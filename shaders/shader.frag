#version 450

//shader input
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;


//output write
layout (location = 0) out vec4 outFragColor;

/*
layout (std430, set = 0, binding = 1) buffer Lights{
    vec3 lightingPosition;
    float Ambient;
};
*/

layout (set = 1, binding = 0) uniform sampler2D displayTexture;


void main()
{
    vec3 LightPosition = vec3(0.0, 10.0, 0.0);
    float ambient = 0.5f;
    outFragColor = texture(displayTexture, inUV);
    outFragColor.w = 1.0;
}
