declare float @__ocml_fabs_f32(float)
define amdgpu_kernel void @k(float addrspace(1)* %p) {
entry:
  %v = load float, float addrspace(1)* %p
  %r = call float @__ocml_fabs_f32(float %v)
  store float %r, float addrspace(1)* %p
  ret void
}
