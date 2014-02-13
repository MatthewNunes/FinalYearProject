__kernel void matrix_mult(__global float4 *a_mat, 
	  __global float4 *b_mat, __global float *c_mat) {

   float sum;

   int num_rows = get_global_size(0);
   int vectors_per_row = num_rows/4;
   int start = get_global_id(0) * vectors_per_row;
   a_mat += start;
   c_mat += start*4;

   for(int i=0; i<num_rows; i++) {
	  sum = 0.0f;
	  for(int j=0; j<vectors_per_row; j++) {
		 sum += dot(a_mat[j], b_mat[i * vectors_per_row + j]);
	  }
	  c_mat[i] = sum;
   }   
}