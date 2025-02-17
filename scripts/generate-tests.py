import duckdb
import os
import requests
import tarfile, io
import tempfile
import tarfile
import zstandard
from pathlib import Path

DUCKPGQ_INSTALL_COMMAND = "force install 'scripts/duckpgq.duckdb_extension'"

PAIR_LIST = [1, 10, 100, 1000, 2048, 4096, 8192, 16384, 16384 * 2]
SCALE_FACTORS = [1, 3, 10, 30]#, 100, 300, 1000]
OPERATIONS = ["iterativelength"]#, "shortestpath"]

def extract_zst(archive: Path, out_path: Path):
    archive = Path(archive).expanduser()
    out_path = Path(out_path).expanduser().resolve()
    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            dctx.copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path, filter="fully_trusted")

def generate_pair_table(conn, num_pairs, seed):
    conn.execute(f"select setseed({seed})")  # Make repeatable experiments

    conn.execute(f"""create or replace table snb_pairs as (
           select src, dst
           from (select a.rowid as src from nodes a),
                (select b.rowid as dst from nodes b)
           using sample reservoir({num_pairs} rows) repeatable (300)
        );""")

def csr_get_query():
    return f"""
    SELECT CREATE_CSR_EDGE(
                0,
                (SELECT count(a.id) FROM nodes a),
                CAST (
                    (SELECT sum(CREATE_CSR_VERTEX(
                                0,
                                (SELECT count(a.id) FROM nodes a),
                                sub.dense_id,
                                sub.cnt)
                                )
                    FROM (
                        SELECT a.rowid as dense_id, count(k.src) as cnt
                        FROM nodes a
                        LEFT JOIN edges k ON k.src = a.id
                        GROUP BY a.rowid) sub
                    )
                AS BIGINT),
                (select count(*) from edges k JOIN nodes a on a.id = k.src JOIN nodes c on c.id = k.dst),
                a.rowid,
                c.rowid,
                k.rowid) as temp
        FROM edges k
        JOIN nodes a on a.id = k.src
        JOIN nodes c on c.id = k.dst
    """

def path_finding_csr_query(graph_operator):
    return f"""WITH cte1 AS (
        SELECT CREATE_CSR_EDGE(
                0,
                (SELECT count(a.id) FROM nodes a),
                CAST (
                    (SELECT sum(CREATE_CSR_VERTEX(
                                0,
                                (SELECT count(a.id) FROM nodes a),
                                sub.dense_id,
                                sub.cnt)
                                )
                    FROM (
                        SELECT a.rowid as dense_id, count(k.src) as cnt
                        FROM nodes a
                        LEFT JOIN edges k ON k.src = a.id
                        GROUP BY a.rowid) sub
                    )
                AS BIGINT),
                (select count(*) from edges k JOIN nodes a on a.id = k.src JOIN nodes c on c.id = k.dst),
                a.rowid,
                c.rowid,
                k.rowid) as temp
        FROM edges k
        JOIN nodes a on a.id = k.src
        JOIN nodes c on c.id = k.dst
    ) SELECT src as source, dst as destination, {graph_operator}(0, (select count(*) from nodes), snb_pairs.src, snb_pairs.dst) as path
            FROM snb_pairs, (select count(cte1.temp) * 0 as temp from cte1) __x
            WHERE __x.temp * 0 = 0;
    """

def get_data(scale, extension):
    columns = '{"id": "BIGINT"}' if extension == "v" else '{ "src": "BIGINT", "dst": "BIGINT", "cost": "BIGINT"}'
    sep = '' if extension == "v" else 'sep = " ",'
    table_name = 'nodes' if extension == "v" else "edges"
    return f"""CREATE TABLE {table_name} AS FROM read_csv("../duckpgq-experiments/data/snb-bi-sf{scale}.{extension}", {sep} columns = { columns })"""

def download_test_data():
    output_dir = './data/snb'
    try:
        os.mkdir(output_dir)
    except Exception:
        pass
    for scale_factor in SCALE_FACTORS:
        if not os.path.exists(f'{output_dir}/bi-sf{scale_factor}.v'):
            continue
        print(f"downloading asset: {output_dir}/bi-sf{scale_factor}")

        response = requests.get(
            f'https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/duckpgq-experiments/snb-bi/snb-bi-sf{scale_factor}.tar.zst',
        )
        with open(f'{output_dir}/bi-sf{scale_factor}.tar.zst', "wb") as file:
            file.write(response.content)
        extract_zst(f'{output_dir}/bi-sf{scale_factor}.tar.zst', './data/snb/')
        os.remove(f'{output_dir}/bi-sf{scale_factor}.tar.zst')

def generate_ground_truth():
    for scale_factor in SCALE_FACTORS:
        conn = duckdb.connect(config = { "allow_unsigned_extensions": "true" })
        conn.execute("")
        conn.execute("load duckpgq")

        output_dir = f'./data/{scale_factor}'
        try:
            os.mkdir(output_dir)
        except Exception:
            pass
        print(f"loading scale factor {scale_factor} data")
        # loads in data from files
        conn.execute(get_data(scale_factor, 'v'))
        conn.execute(get_data(scale_factor, 'e'))

        # creates csr
        conn.execute(csr_get_query()).df()
        # fetch csr
        vertex_result = conn.execute("SELECT csrv FROM get_csr_v(0)").fetchnumpy()
        edge_result = conn.execute("SELECT csre FROM get_csr_e(0)").fetchnumpy()

        # delete csr
        conn.execute("SELECT delete_csr(0)")
        # output CSR to file
        try:
            with open(f'{output_dir}/v.bin', "xb") as output:
                output.write(vertex_result['csrv'].astype('uint32'))
        except:
            pass
        try:
            with open(f'{output_dir}/e.bin', "xb") as output:
                output.write(edge_result['csre'].astype('uint32'))
        except:
            pass
        # generate ground truth
        for pair in PAIR_LIST:
            if not os.path.exists(f'{output_dir}/{pair}-src.bin'):
                print(f"generating pair {pair} for {scale_factor}")
                # generate pairs
                generate_pair_table(conn, pair, 0.42)
                # get pairs
                result = conn.execute("SELECT * FROM snb_pairs").fetchnumpy()
                try:
                    # write src column to disk
                    with open (f'{output_dir}/{pair}-src.bin', "xb") as output:
                        output.write(result['src'].astype('uint32'))
                except:
                    pass
                try:
                    # write dst column to disk
                    with open (f'{output_dir}/{pair}-dst.bin', "xb") as output:
                        output.write(result['dst'].astype('uint32'))
                except:
                    pass


            # generate ground truth
            for operator in OPERATIONS:
                # do path finding operation
                if os.path.exists(f'{output_dir}/{pair}-{operator}-truth.csv'):
                    continue
                print(f"generating ground truth for {operator}")
                data = conn.execute(path_finding_csr_query(operator)).df()
                data = duckdb.sql(f"COPY (SELECT * FROM data WHERE path IS NOT NULL) TO '{output_dir}/{pair}-{operator}-truth.csv'")

        conn.execute("DROP TABLE nodes")
        conn.execute("DROP TABLE edges")

download_test_data()
generate_ground_truth()
