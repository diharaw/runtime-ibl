layout(location = 0) in vec3 VS_IN_Position;
layout(location = 1) in vec2 VS_IN_Texcoord;
layout(location = 2) in vec3 VS_IN_Normal;
layout(location = 3) in vec3 VS_IN_Tangent;
layout(location = 4) in vec3 VS_IN_Bitangent;

out vec3 PS_IN_FragPos;
out vec3 PS_IN_Normal;
out vec2 PS_IN_TexCoord;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Projection;

void main()
{
    vec4 world_pos = u_Model * vec4(VS_IN_Position, 1.0f);
    PS_IN_FragPos  = world_pos.xyz;
    PS_IN_TexCoord = VS_IN_Texcoord;

    mat3 model_mat = mat3(u_Model);

    PS_IN_Normal = normalize(model_mat * VS_IN_Normal);

    gl_Position = u_Projection * u_View * world_pos;
}
