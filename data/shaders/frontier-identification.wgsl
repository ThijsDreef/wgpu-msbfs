@group(0)
@binding(0)
var<storage, read_write> jfq: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> jfq_length: u32;

@group(1)
@binding(0)
var<storage> dst: array<u32>;

@group(1)
@binding(1)
var<storage, read_write> path_length: array<u32>;

@group(1)
@binding(2)
var<uniform> iteration: u32;

const amount_of_searches: u32 = 4;
@group(1)
@binding(3)
var<storage, read_write> mask: array<u32>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<u32>;


fn check_path(vertex: u32, diff: u32, offset: u32) {
  var local_diff = diff;
  var length: u32 = countOneBits(local_diff);
  for (var j = 0u; j < length; j++) {
    var index: u32 = countTrailingZeros(local_diff);
    var path_index = (index) + (32u * offset);
    if (dst[path_index] == vertex) {
      path_length[path_index] = iteration;
      mask[offset] ^= 1u << index;
    }
    local_diff ^= (1u << index);
  }
}

fn topdown() {
  var vertices : u32 = arrayLength(&jfq);
  jfq_length = 0u;
  for (var i : u32 = 0; i < vertices; i++) {
    var offset = i * amount_of_searches;
    var diff : vec4<u32> = vec4u(
       (bsa[offset] ^ bsak[offset]) & mask[0],
       (bsa[offset + 1] ^ bsak[offset + 1]) & mask[1],
       (bsa[offset + 2] ^ bsak[offset + 2]) & mask[2],
       (bsa[offset + 3] ^ bsak[offset + 3]) & mask[3],
    );



    bsak[offset] |= bsa[offset];
    bsak[offset + 1] |= bsa[offset + 1];
    bsak[offset + 2] |= bsa[offset + 2];
    bsak[offset + 3] |= bsa[offset + 3];

    if (diff.x == 0 && diff.y == 0 && diff.z == 0 && diff.w == 0) {
      continue;
    }

    jfq[jfq_length] = i;
    jfq_length++;

    check_path(i, diff.x, 0u);
    check_path(i, diff.y, 1u);
    check_path(i, diff.z, 2u);
    check_path(i, diff.w, 3u);
  }
}

fn bottomup() {
  var bsal : u32 = arrayLength(&bsa);
  jfq_length = 0u;
  for (var i : u32 = 0; i < bsal; i++) {
    var value : u32 = select(1u, 0u, ((~bsak[i]) & 1) == 0);
    jfq[jfq_length] = i;
    jfq_length += value;
    bsak[i] |= bsa[i];
  }
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  topdown();
}
