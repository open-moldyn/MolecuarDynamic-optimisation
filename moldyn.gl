// %%VARIABLE%% will be replaced with consts by python code

#version 430


#define X %%X%%
#define Y %%Y%%
#define Z %%Z%%
#define RCUT %%RCUT%%
#define EPSILON %%EPSILON%%
#define SIGMA %%SIGMA%%
#define SHIFTX %%SHIFTX%%
#define SHIFTY %%SHIFTY%%
#define LENGTHX %%LENGTHX%%
#define LENGTHY %%LENGTHY%%


layout(local_size_x=X, local_size_y=1, local_size_z=1) in;
layout (std430, binding=0) buffer in_0
{
    vec2 inxs[];
};

layout (std430, binding=1) buffer out_0
{
    vec2 outxs[];
};

layout (std430, binding=2) buffer out_1
{
    float outes[];
};

layout (std430, binding=3) buffer out_2
{
    float outms[];
};

float force(float dist) {
	const float p=pow(SIGMA/dist,6);
	return (-4.0*EPSILON*(6.0*p-12.0*p*p))/(dist*dist);
}

float energy(float dist) {
	const float p=pow(SIGMA/dist,6);
	return EPSILON*(4.0*(p*p-p)+127.0/4096.0);
}

void main()
{
	const int x = int(gl_GlobalInvocationID.x);
	const vec2 pos = inxs[x];
	
	vec2 f = vec2(0.0,0.0);
	float e = 0.0;
	float m = 0.0;
	
	for(int i=0;i<X;i++) {
		if(i!=x) {
			vec2 distxy = pos - inxs[i];
			
			if(distxy.x<(-SHIFTX)) {
				distxy.x+=LENGTHX;
			}
			if(distxy.x>SHIFTX) {
				distxy.x-=LENGTHX;
			}
			
			if(distxy.y<(-SHIFTY)) {
				distxy.y+=LENGTHY;
			}
			if(distxy.y>SHIFTY) {
				distxy.y-=LENGTHY;
			}
			
			
			float dist = length(distxy);
			if(dist<RCUT) {
				f+=force(dist)*distxy;
				e+=energy(dist);
				m+=1.0;
			}
		}
	}
	
	outxs[x] = f;
	outes[x] = e;
	outms[x] = m;
}