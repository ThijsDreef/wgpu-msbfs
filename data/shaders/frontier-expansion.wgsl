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

const x_size = 48u;
const y_size = 16u;

fn topdown(
  local_id: vec3<u32>,
  invocation_size: vec3<u32>,
  invocation_id: vec3<u32>,
) {
  var id = invocation_id.x;
  var bsa_offset = id * (arrayLength(&v) / 2);
  var jfq_length = search_info[id].jfq_length;
  for (var i : u32 = local_id.x; i < jfq_length; i += x_size) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex] + local_id.y;
    var end: u32 = v[vertex + 1];
    for (; start < end; start += y_size) {
      var edge = e[start];
      atomicOr(&bsak[bsa_offset + edge], bsa[vertex + bsa_offset]);
    }
  }
}

fn bottomup(
  local_id: vec3<u32>,
  invocation_size: vec3<u32>,
  invocation_id: vec3<u32>,
) {
  var id = invocation_id.x;
  var v_offset = arrayLength(&v) / 2;
  var e_offset = arrayLength(&e) / 2;
  var bsa_offset = id * v_offset;
  var jfq_length = search_info[id].jfq_length;
  for (var i : u32 = local_id.x; i < jfq_length; i += x_size) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = e_offset + v[vertex + v_offset] + local_id.y;
    var end: u32 = e_offset + v[vertex + v_offset + 1];
    var current = bsak[bsa_offset + vertex];
    for (; start < end; start += y_size) {
      current |= bsa[bsa_offset + e[start]];
    }
    atomicOr(&bsak[bsa_offset + vertex], current);
  }
}

@compute
@workgroup_size(x_size, y_size, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) invocation_size: vec3<u32>,
  @builtin(workgroup_id) invocation_id: vec3<u32>,
) {
  var id = invocation_id.x;
  var iteration = search_info[id].iteration;

  if (iteration >= 4u && iteration <= 5u) {
    bottomup(local_id, invocation_size, invocation_id);
  } else {
    topdown(local_id, invocation_size, invocation_id);
  }

}
