struct SearchInfo {
  iteration: u32,
  mask: u32,
  jfq_length: u32,
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
var<storage> search_info: array<SearchInfo, 64>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<atomic<u32>>;

@compute
@workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>,
  @builtin(workgroup_id) invocation_id: vec3<u32>,
) {
  var id = local_id.x;
  var bsa_offset = id * arrayLength(&v);
  for (var i : u32 = invocation_id.x; i < search_info[id].jfq_length; i += invocation_size.x) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsak[bsa_offset + edge], bsa[bsa_offset + vertex]);
    }
  }
}
