#version 450

//shader input
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inFragPos;
layout (location = 4) in vec3 inCameraPosition;




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
    // Hard coded light position
    vec3 lightPosition = vec3(10.0, 20.0, 0.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float lightAmbient = 0.1f;
    float lightSpecularStrength = 0.5;

    // Ambient calculation
    vec3 ambient = lightAmbient * lightColor;

    // Diffuse calculation
    vec3 norm = normalize(inNormal);
    vec3 lightDir = normalize(lightPosition - inFragPos);

    float diff = max(dot(norm, lightDir), 0.0);

    vec3 diffuse = diff * lightColor;

    // Specular calculation
    vec3 viewDir = normalize(inCameraPosition - inFragPos);
    vec3 reflectDir = reflect(-lightDir, norm);

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    vec3 specular = lightSpecularStrength * spec * lightColor;

    // Combine ambient diffuse and specular and texture
    vec4 result = vec4(ambient + diffuse + specular, 1.0) * texture(displayTexture, inUV);
    outFragColor = result;
}
