@group(0)
@binding(0)
var<storage> jfq: array<u32>;

@group(0)
@binding(1)
var <storage> jfq_length: u32;

@group(1)
@binding(0)
var<storage> v: array<u32>;

@group(1)
@binding(1)
var<storage> e: array<u32>;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<atomic<u32>>;

fn topdown(id: u32, stride: u32) {
  for (var i : u32 = id; i < jfq_length; i += 64u * stride) {
    var vertex = jfq[i];
    // If we want to avoid copying this is required.
    bsak[vertex] |= bsa[vertex];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsak[edge], bsa[vertex]);
    }
  }
}

fn bottomup(id: u32, stride: u32) {
  for (var i : u32 = id; i < jfq_length; i += 64u * stride) {
    var frontier = jfq[i];
    var neighbour: u32 = v[frontier];
    var end: u32 = v[frontier + 1];
    for (; neighbour < end; neighbour++) {
      var edge = e[neighbour];
      // This requires usage of the mask
      bsak[frontier] |= bsa[edge];
    }
  }
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups)       num_workgroups: vec3<u32>) {
  topdown(global_id.x, num_workgroups.x);
}
