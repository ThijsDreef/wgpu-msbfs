@group(0)
@binding(0)
var<storage> v: array<u32>;
@group(0)
@binding(1)
var<storage> e: array<u32>;
@group(0)
@binding(2)
var<storage, read_write> path_length: array<u32>;
@group(0)
@binding(3)
var<storage, read_write> jfq: array<u32>;
@group(0)
@binding(4)
var<storage> dst: array<u32>;
@group(0)
@binding(5)
var<storage, read_write> bsa: array<atomic<u32>>;
@group(0)
@binding(6)
var<storage, read_write> bsak: array<atomic<u32>>;

var<workgroup> mask: u32;
var<workgroup> iteration: u32;
var<workgroup> jfq_length: u32;

fn expand(bsa_offset: u32, id: u32) {
  for (var i : u32 = id; i < jfq_length; i += 64u) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsak[edge + bsa_offset], atomicLoad(&bsa[vertex + bsa_offset]));
    }
  }
}

fn identify(bsa_offset: u32, dst_offset: u32) {
  var vertices_length = arrayLength(&v);
  jfq_length = 0u;
  for (var i : u32 = 0; i < vertices_length; i++) {
    var diff : u32 = (atomicLoad(&bsa[i + bsa_offset]) ^ atomicLoad(&bsak[i + bsa_offset])) & mask;
    if (diff == 0) { continue; }
    atomicOr(&bsak[i + bsa_offset], atomicLoad(&bsa[i + bsa_offset]));
    jfq[bsa_offset + jfq_length] = i;
    jfq_length++;
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[dst_offset + index] == i) {
        path_length[dst_offset + index] = iteration;
        mask &= ~(1u << index);
      }
      diff &= ~(1u << index);
    }
  }
}


fn expand_p(bsa_offset: u32, id: u32) {
  for (var i : u32 = id; i < jfq_length; i += 64u) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsa[edge + bsa_offset], atomicLoad(&bsak[vertex + bsa_offset]));
    }
  }
}

fn identify_p(bsa_offset: u32, dst_offset: u32) {
  var vertices_length = arrayLength(&v);
  jfq_length = 0u;
  for (var i : u32 = 0; i < vertices_length; i++) {
    var diff : u32 = (atomicLoad(&bsak[i + bsa_offset]) ^ atomicLoad(&bsa[i + bsa_offset])) & mask;
    if (diff == 0) { continue; }
    atomicOr(&bsa[i + bsa_offset], atomicLoad(&bsak[i + bsa_offset]));
    jfq[bsa_offset + jfq_length] = i;
    jfq_length++;
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[dst_offset + index] == i) {
        path_length[dst_offset + index] = iteration;
        mask &= ~(1u << index);
      }
      diff &= ~(1u << index);
    }
  }
}


@compute
@workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) invocation: vec3<u32>) {
  var bsa_offset = invocation.x * arrayLength(&v);
  var dst_offset = invocation.x * 32u;
  mask = ~0u;
  iteration = 0u;
  jfq_length = 1u;
  for (var i = 0u; i < arrayLength(&v) / 2; i++) {
    expand(bsa_offset, local_id.x);
    workgroupBarrier();
    if (local_id.x == 0 && (jfq_length > 0)) {
      identify(bsa_offset, dst_offset);
      iteration++;
    }
    workgroupBarrier();
    expand_p(bsa_offset, local_id.x);
    workgroupBarrier();
    if (local_id.x == 0 && (jfq_length > 0)) {
      identify_p(bsa_offset, dst_offset);
      iteration++;
    }
    workgroupBarrier();
  }

}
