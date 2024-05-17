#version 450
#extension GL_EXT_buffer_reference : require

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out vec3 outFragPos;
layout (location = 4) out vec3 outCameraPosition;


struct Vertex {

	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer{
	Vertex vertices[];
};

layout (std140, set = 0, binding = 0) uniform render_data{
	mat4 view;
	mat4 projection;
	mat4 view_proj;
	vec4 ambient_color;
	vec4 sunlight_direction;
	vec4 sunlight_color;
	vec4 viewPosition;
};

//push constants block
layout( push_constant ) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main()
{
	//load vertex data from device adress
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];

	//output data
	gl_Position = projection * view * PushConstants.render_matrix * vec4(v.position, 1.0f);
	outColor = v.color.xyz;
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
	outNormal = mat3(transpose(inverse(PushConstants.render_matrix))) * v.normal;
	outCameraPosition = vec3(viewPosition);
	outFragPos = vec3(PushConstants.render_matrix * vec4(v.position, 1.0));
}
