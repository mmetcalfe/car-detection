#version 150

in vec3 position;
in vec3 normal;

out vec4 Normal;

uniform mat4 model;
// uniform mat4 view;
uniform mat4 proj;

void main() {
    // gl_Position = proj * view * model * vec4(position, 1.0);
    gl_Position = proj * model * vec4(position, 1.0);
    // gl_Position = vec4((proj * model * vec4(position, 1.0)).xy, 0.5, 1.0);
    // gl_Position = proj * model * vec4(
    //       position.x - 0.0
    //     , position.y - 0.0
    //     , position.z - 0.0
    //     , 1.0);
    // gl_Position = vec4(position, 1.0);
    Normal = vec4(normal, 0.0);
}
