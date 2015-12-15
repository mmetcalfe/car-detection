#version 150

// uniform vec3 colDiffuse;

out vec4 outColor;

in vec4 Normal;

void main() {
    outColor = vec4(1.0,1.0, 0.0, 1.0);
    // outColor = vec4(colDiffuse, 1.0);
}
