// %%VARIABLE%% will be replaced with consts by python code

#version 430


#define X %%X%%
#define NPART %%NPART%%
#define N_A %%N_A%%
#define RCUT %%RCUT%%

#define EPSILONA %%EPSILONA%%
#define EPSILONB %%EPSILONB%%
#define EPSILONAB %%EPSILONAB%%

#define SIGMAA %%SIGMAA%%
#define SIGMAB %%SIGMAB%%
#define SIGMAAB %%SIGMAAB%%

#define SHIFTX %%SHIFTX%%
#define SHIFTY %%SHIFTY%%
#define LENGTHX %%LENGTHX%%
#define LENGTHY %%LENGTHY%%


#define RCUT2 RCUT*RCUT

layout (local_size_x=X, local_size_y=1, local_size_z=1) in;

layout (std430, binding=0) buffer in_0
{
    vec2 inxs[];
};

layout (std430, binding=2) buffer out_0
{
    vec2 outfs[];
};


layout (std430, binding=3) buffer out_1
{
    float outes[];
};

layout (std430, binding=4) buffer out_2
{
    float outms[];
};

layout (std430, binding=5) buffer in_params
{
    uint inparams[];
};

float force(float dist,float p, float epsilon) {
	return (-4.0*epsilon*(6.0*p-12.0*p*p))/(dist*dist);
}

float energy(float dist,float p, float epsilon) {
	return epsilon*(4.0*(p*p-p)+127.0/4096.0);
}

void iterate(vec2 pos, uint a, uint b, float epsilon, float sigma) {
	const uint x = gl_GlobalInvocationID.x;

	for (uint i=a;i<b;i++) {
		if (i!=x) {
			vec2 distxy = pos - inxs[i];

			if (distxy.x<(-SHIFTX)) {
				distxy.x+=LENGTHX;
			}
			if (distxy.x>SHIFTX) {
				distxy.x-=LENGTHX;
			}

			if (distxy.y<(-SHIFTY)) {
				distxy.y+=LENGTHY;
			}
			if (distxy.y>SHIFTY) {
				distxy.y-=LENGTHY;
			}


			/* Ce test accélère d'environ 15%, puisqu'on saute les étapes de multipication+somme du calcul de distance
			 * pour voir si on est dans la sphère
			 */
			if(abs(distxy.x)<RCUT && abs(distxy.y)<RCUT) {

				float dist = length(distxy);

				if (dist<RCUT) {
					const float p=pow(sigma/dist, 6);

					outfs[x] += force(dist, p, epsilon)*distxy;
					outes[x] += energy(dist, p, epsilon);
					outms[x] += 1.0;
				}
			}
		}
	}
}

void main()
{
	const uint x = gl_GlobalInvocationID.x;
	const vec2 pos = inxs[x];

	outfs[x] = vec2(0.0);
	outes[x] = 0.0;
	outms[x] = 0.0;

	if(x < NPART) {

		if(x < N_A) {
			iterate(pos,0,N_A,EPSILONA,SIGMAA);
			iterate(pos,N_A,NPART,EPSILONAB,SIGMAAB);
		} else {
			iterate(pos,0,N_A,EPSILONAB,SIGMAAB);
			iterate(pos,N_A,NPART,EPSILONB,SIGMAB);
		}

	}
}