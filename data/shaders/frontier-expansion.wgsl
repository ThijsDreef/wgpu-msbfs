struct SearchInfo {
  iteration: u32,
  mask: u32,
  jfq_length: u32,
  padding: u32,
};

@group(0)
@binding(0)
var<storage> v: array<u32>;

@group(0)
@binding(1)
var<storage> e: array<u32>;

@group(1)
@binding(0)
var<storage> jfq: array<u32>;

@group(1)
@binding(1)
var<storage> search_info: array<SearchInfo>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<atomic<u32>>;

var<workgroup> index: atomic<u32>;

@compute
@workgroup_size(32, 8, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>,
  @builtin(workgroup_id) invocation_id: vec3<u32>,
) {
  var id = invocation_id.x + (invocation_id.z * 64u);
  var bsa_offset = id * arrayLength(&v);
  var jfq_length = search_info[id].jfq_length;
  for (var i : u32 = local_id.x; i < jfq_length; i += 32u) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex] + local_id.y;
    var end: u32 = v[vertex + 1];
    for (; start < end; start += 8u) {
      var edge = e[start];
      atomicOr(&bsak[bsa_offset + edge], bsa[bsa_offset + vertex]);
    }
  }
}
