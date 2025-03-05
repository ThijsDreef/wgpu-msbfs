@group(0)
@binding(0)
var<storage, read_write> jfq: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> jfq_length: u32;

@group(1)
@binding(0)
var<storage> dst: array<u32>;

@group(1)
@binding(1)
var<storage, read_write> path_length: array<u32>;

@group(1)
@binding(2)
var<uniform> iteration: u32;

@group(1)
@binding(3)
var<storage, read_write> mask: u32;

@group(2)
@binding(0)
var<storage> bsa: array<u32>;

@group(2)
@binding(1)
var<storage, read_write> bsak: array<u32>;

fn topdown() {
  var bsal : u32 = arrayLength(&bsa);
  jfq_length = 0u;
  for (var i : u32 = 0; i < bsal; i++) {
    var diff : u32 = (bsa[i] ^ bsak[i]) & mask;
    if (diff == 0) {
      continue;
    }
    bsak[i] |= bsa[i];
    jfq[jfq_length] = i;
    jfq_length++;
    var length: u32 = countOneBits(diff);
    for (var j = 0u; j < length; j++) {
      var index: u32 = countTrailingZeros(diff);
      if (dst[index] == i) {
        path_length[index] = iteration;
        mask ^= 1u << index;
      }
      diff &= ~(1u << index);
    }

  }
}

fn bottomup() {
  var bsal : u32 = arrayLength(&bsa);
  jfq_length = 0u;
  for (var i : u32 = 0; i < bsal; i++) {
    var value : u32 = select(1u, 0u, ((~bsak[i]) & 1) == 0);
    jfq[jfq_length] = i;
    jfq_length += value;
    bsak[i] |= bsa[i];
  }
}

@compute
@workgroup_size(1)
fn main() {
  topdown();
}
