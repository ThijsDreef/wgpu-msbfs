struct SearchInfo {
  iteration: u32,
  jfq_length: atomic<u32>,
  last_jfq: u32,
  mask: array<atomic<u32>, 32>,

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

const workgroup_size : u32 = 32;

var<workgroup> prefix: array<u32, workgroup_size>;

@compute
@workgroup_size(workgroup_size)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) invocation: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>) {
  var c_mask = ~atomicLoad(&search_info[0].mask[local_id.x]);
  var iteration = search_info[0].iteration;
  var dst_offset = local_id.x * workgroup_size;
  for (var i : u32 = invocation.y; i < arrayLength(&jfq); i += invocation_size.y) {
    var bsa_offset = i * workgroup_size + local_id.x;
    var diff : u32 = (bsa[bsa_offset] ^ bsak[bsa_offset]) & c_mask;
    prefix[local_id.x] = diff;
    bsak[bsa_offset] |= bsa[bsa_offset];
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[dst_offset + index] == i) {
        path_length[dst_offset + index] = iteration;
        atomicOr(&search_info[0].mask[local_id.x], (1u << index));
        c_mask ^= 1u << index;
      }
      diff ^= (1u << index);
    }
    storageBarrier();
    // TODO: Why is this faster then letting all threads execute this.
    if (local_id.x == 0u) {
      var acc = 0u;
      for (var j : u32 = 0u; j < 32; j++) {
        acc |= prefix[j];
      }
      if (acc > 0) {
        jfq[atomicAdd(&search_info[0].jfq_length, 1u)] = i;
      }
    }
    workgroupBarrier();
  }
}
