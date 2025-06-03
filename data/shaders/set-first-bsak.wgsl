struct SearchInfo {
  offset: u32,
  iteration: u32,
  jfq_length: u32,
  last_jfq: u32,
  mask: array<u32, 32>,
};

@group(0)
@binding(0)
var<storage, read> info : SearchInfo;
@group(0)
@binding(1)
var<storage, read> src : array<u32>;
@group(0)
@binding(2)
var<storage, read_write> bsak : array<atomic<u32>>;

@compute
@workgroup_size(32, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) invocation: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>
) {
  if (local_id.x + invocation.x * invocation_size.x >= arrayLength(&src)) {
    return;
  }

  var index = local_id.x + invocation.x * invocation_size.x;
  // Copy dst -> target_dst
  var temp = src[index + info.offset] * invocation_size.x + invocation.x;
  // set BSAK
  atomicOr(&bsak[temp], 1u << local_id.x);
}
