struct SearchInfo {
  iteration: u32,
  mask: u32,
  jfq_length: u32,
};

@group(0)
@binding(0)
var<storage, read_write> jfq: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> search_info: array<SearchInfo, 64>;

@group(1)
@binding(0)
var<storage> dst: array<u32, 32 * 64>;

@group(1)
@binding(1)
var<storage, read_write> path_length: array<u32, 32 * 64>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<u32>;

var<workgroup> jfq_length: atomic<u32>;
var<workgroup> mask: atomic<u32>;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) invocation: vec3<u32>) {
  var id = invocation.x;
  var bsa_offset = id * (arrayLength(&jfq) / 64u);
  var dst_offset = id * 32u;
  var maskl = ~search_info[id].mask;
  var iteration = search_info[id].iteration;
  atomicStore(&jfq_length, 0u);
  atomicStore(&mask, maskl);
  workgroupBarrier();
  if (search_info[id].jfq_length == 0 && iteration > 0) {return;}
  for (var i : u32 = local_id.x; i < arrayLength(&jfq) / 64; i += 256) {
    var diff : u32 = (bsa[bsa_offset + i] ^ bsak[bsa_offset + i]) & maskl;
    if (diff == 0) { continue; }
    bsak[bsa_offset + i] |= bsa[bsa_offset + i];
    var temp = atomicAdd(&jfq_length, 1u);
    jfq[bsa_offset + temp] = i;
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[dst_offset + index] == i) {
        path_length[dst_offset + index] = iteration;
        atomicAnd(&mask, ~(1u << index));
      }
      diff &= ~(1u << index);
    }
  }
  search_info[id].mask = ~atomicLoad(&mask);
  search_info[id].jfq_length = atomicLoad(&jfq_length);
  search_info[id].iteration = iteration + 1;
}
