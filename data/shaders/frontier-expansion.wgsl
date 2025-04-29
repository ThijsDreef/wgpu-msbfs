struct SearchInfo {
  iteration: u32,
  mask: u32,
  jfq_length: u32,
  last_jfq: u32,
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
var<storage, read_write> search_info: array<SearchInfo>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<atomic<u32>>;

@compute
@workgroup_size(128, 8, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>,
  @builtin(workgroup_id) invocation_id: vec3<u32>,
) {
  var id = invocation_id.x;
  if (local_id.x == 0u && local_id.y == 0u && invocation_id.y == 0u) {
    search_info[id].iteration += 1;
    search_info[id].last_jfq = search_info[id].jfq_length;
  }
  var bsa_offset = id * arrayLength(&v);
  var jfq_length = search_info[id].jfq_length;
  for (var i : u32 = local_id.x + invocation_id.y * 128u; i < jfq_length; i += 128u * invocation_size.y) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex] + local_id.y;
    var end: u32 = v[vertex + 1];
    for (; start < end; start += 8) {
      var edge = e[start];
      atomicOr(&bsak[bsa_offset + edge], bsa[vertex + bsa_offset]);
    }
  }


}
