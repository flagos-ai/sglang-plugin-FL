#!/usr/bin/env bash
# Source this file before starting SGLang or running the fixed-tensor probe.
unset SGLANG_PLATFORM SGLANG_FL_FLAGOS_BLACKLIST
export SGLANG_PLUGINS=sglang_fl USE_C_EXTENSION=0 USE_FLAGGEMS=1
export SGLANG_FL_PRESERVE_NATIVE_RUNTIME=1 SGLANG_FL_GLM52_MUSA=1
export SGLANG_FL_OOT_ENABLED=1 SGLANG_FL_OOT_WHITELIST=RMSNorm
export SGLANG_FL_PER_OP="rms_norm=flagos" SGLANG_FL_STRICT=1
export SGLANG_FL_FLAGOS_WHITELIST=add,mul,add_,arange,arange_start,bitwise_and_scalar,bitwise_and_scalar_tensor,bitwise_and_tensor,bitwise_not,cat,cat_out,clamp,clamp_tensor,fill_scalar_,fill_tensor_,floor_divide,ge,ge_scalar,lt,lt_scalar,masked_fill_,sub,zero_,_index_put_impl_,index_select,cumsum,cumsum_out,layer_norm,bmm,bmm_out,eq_scalar,eq,gt_scalar,gt,ne_scalar,ne,where_self,where_self_out,max,max_dim,min,min_dim,index_add_,scatter_
