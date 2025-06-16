struct SearchInfo {
  offset: u32,
  iteration: u32,
  jfq_length: u32,
  last_jfq: u32,
  mask: array<u32, 32>,
};

@group(0)
@binding(0)
var<storage> v: array<u32>;

@group(0)
@binding(1)
var<storage> e: array<u32>;

@group(1)
@binding(0)
var<storage, read_write> jfq: array<u32>;

@group(1)
@binding(1)
var<storage, read_write> search_info: SearchInfo;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<atomic<u32>>;

@compute
@workgroup_size(32, 8, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>,
  @builtin(workgroup_id) invocation_id: vec3<u32>,
) {
  if (local_id.x == 0u && local_id.y == 0u && invocation_id.y == 0u) {
    search_info.iteration += 1;
    search_info.last_jfq = search_info.jfq_length;
  }

  var v_offset = arrayLength(&v) / 2;
  var e_offset = arrayLength(&e) / 2;
  var jfq_length = search_info.jfq_length;
  for (var i : u32 = invocation_id.y; i < jfq_length; i += invocation_size.y) {
    var vertex = jfq[i];
    var start: u32 = e_offset + v[vertex + v_offset] + local_id.y;
    var end: u32 = e_offset + v[vertex + v_offset + 1];
    var current = atomicLoad(&bsak[vertex * 32 + local_id.x]);
    for (; start < end; start += 8) {
      current |= bsa[e[start] * 32 + local_id.x];
    }
    atomicOr(&bsak[vertex * 32 + local_id.x], current);
  }
}
