struct SearchInfo {
  iteration: u32,
  mask: atomic<u32>,
  jfq_length: atomic<u32>,
  last_jfq: u32,
};

@group(0)
@binding(0)
var<storage, read_write> jfq: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> search_info: array<SearchInfo>;

@group(1)
@binding(0)
var<storage> dst: array<u32>;

@group(1)
@binding(1)
var<storage, read_write> path_length: array<u32>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<u32>;

@compute
@workgroup_size(256)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) invocation: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>) {

  var id = invocation.x;
  if (search_info[id].iteration > 0 && search_info[id].last_jfq == 0) {
    return;
  }
  var bsa_offset = id * (arrayLength(&jfq) / invocation_size.x);
  var dst_offset = id * 32u;
  var maskl = ~atomicLoad(&search_info[id].mask);
  var iteration = search_info[id].iteration;
  for (var i : u32 = local_id.x + invocation.y * 256u; i < arrayLength(&jfq) / invocation_size.x; i += 256 * invocation_size.y) {
    var diff : u32 = (bsa[bsa_offset + i] ^ bsak[bsa_offset + i]) & maskl;
    if (diff == 0) { continue; }
    bsak[bsa_offset + i] |= bsa[bsa_offset + i];
    var temp = atomicAdd(&search_info[id].jfq_length, 1u);
    jfq[bsa_offset + temp] = i;
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[dst_offset + index] == i) {
        path_length[dst_offset + index] = iteration;
        atomicOr(&search_info[id].mask, (1u << index));
      }
      diff ^= (1u << index);
    }
  }
}
