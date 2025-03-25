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

var<workgroup> mask: atomic<u32>;
var<workgroup> jfq_length: atomic<u32>;

fn expand(bsa_offset: u32, id: u32) {
  var jfq_l = atomicLoad(&jfq_length);
  for (var i : u32 = id; i < jfq_l; i += 64u) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsak[edge + bsa_offset], atomicLoad(&bsa[vertex + bsa_offset]));
    }
  }
}

fn identify(id: u32, bsa_offset: u32, dst_offset: u32, iteration: u32) {
  var vertices_length = arrayLength(&v);
  atomicStore(&jfq_length, 0u);
  workgroupBarrier();
  var mask_l = atomicLoad(&mask);
  for (var i : u32 = id; i < vertices_length; i += 64u) {
    var diff : u32 = (atomicLoad(&bsa[i + bsa_offset]) ^ atomicLoad(&bsak[i + bsa_offset])) & mask_l;
    if (diff == 0) { continue; }
    atomicOr(&bsak[i + bsa_offset], atomicLoad(&bsa[i + bsa_offset]));
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
}


fn expand_p(bsa_offset: u32, id: u32) {
  var jfq_l = atomicLoad(&jfq_length);
  for (var i : u32 = id; i < jfq_l; i += 64u) {
    var vertex = jfq[bsa_offset + i];
    var start: u32 = v[vertex];
    var end: u32 = v[vertex + 1];
    for (; start < end; start++) {
      var edge = e[start];
      atomicOr(&bsa[edge + bsa_offset], atomicLoad(&bsak[vertex + bsa_offset]));
    }
  }
}

fn identify_p(id: u32, bsa_offset: u32, dst_offset: u32, iteration: u32) {
  atomicStore(&jfq_length, 0u);
  workgroupBarrier();
  var vertices_length = arrayLength(&v);
  var mask_l = atomicLoad(&mask);
  for (var i : u32 = id; i < vertices_length; i += 64u) {
    var diff : u32 = (atomicLoad(&bsak[i + bsa_offset]) ^ atomicLoad(&bsa[i + bsa_offset])) & mask_l;
    if (diff == 0) { continue; }
    atomicOr(&bsa[i + bsa_offset], atomicLoad(&bsak[i + bsa_offset]));
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
}


@compute
@workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) invocation: vec3<u32>) {
  var bsa_offset = invocation.x * arrayLength(&v);
  var dst_offset = invocation.x * 32u;
  atomicStore(&mask, ~0u);
  atomicStore(&jfq_length, 0u);
  workgroupBarrier();
  // TODO allow workgroupbariers in tint with possibly non uniform control flow.
  for (var i = 0u; i < 25; i++) {
    expand(bsa_offset, local_id.x);
    workgroupBarrier();
    identify(local_id.x, bsa_offset, dst_offset, i * 2);
    workgroupBarrier();
    expand_p(bsa_offset, local_id.x);
    workgroupBarrier();
    identify_p(local_id.x, bsa_offset, dst_offset, i * 2 + 1);
    workgroupBarrier();

  }

}
