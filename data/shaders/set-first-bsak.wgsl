struct SearchInfo {
  offset: u32,
  iteration: u32,
  mask: u32,
  jfq_length: u32,
  last_jfq: u32,
};

@group(0)
@binding(0)
var<storage, read> info : array<SearchInfo>;
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
  var index = local_id.x + info[invocation.x].offset;

  if (index >= arrayLength(&src)) {
    return;
  }
  var v_size = arrayLength(&bsak) / invocation_size.x;
  var temp = src[index] + v_size * invocation.x;
  atomicOr(&bsak[temp], 1u << local_id.x);
}
