#version 430


in vec2 in_vert;

out vec4 v_color;

uniform vec2 Scale;
uniform int N_A;

void main() {
    if(gl_VertexID < N_A) {
        v_color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        v_color = vec4(0.0, 0.0, 1.0, 1.0);
    }
    gl_Position = vec4(((in_vert * Scale)-1.0), 0.0, 1.0);
}