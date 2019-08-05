#include <atmosphere.glsl>

#define SH_DEBUG

out vec3 PS_OUT_Color;

in vec3 FS_IN_WorldPos;

uniform samplerCube s_Cubemap;
uniform samplerCube s_Prefilter;
uniform sampler2D s_SH;

uniform int   u_Type;
uniform float u_Roughness;
uniform vec3  u_CameraPos;

const float Pi       = 3.141592654;
const float CosineA0 = Pi;
const float CosineA1 = (2.0 * Pi) / 3.0;
const float CosineA2 = Pi * 0.25;

// ------------------------------------------------------------------
// STRUCTURES -------------------------------------------------------
// ------------------------------------------------------------------

struct SH9
{
    float c[9];
};

// ------------------------------------------------------------------

struct SH9Color
{
    vec3 c[9];
};

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

void project_onto_sh9(in vec3 dir, inout SH9 sh)
{
    // Band 0
    sh.c[0] = 0.282095;

    // Band 1
    sh.c[1] = -0.488603 * dir.y;
    sh.c[2] = 0.488603 * dir.z;
    sh.c[3] = -0.488603 * dir.x;

    // Band 2
    sh.c[4] = 1.092548 * dir.x * dir.y;
    sh.c[5] = -1.092548 * dir.y * dir.z;
    sh.c[6] = 0.315392 * (3.0 * dir.z * dir.z - 1.0);
    sh.c[7] = -1.092548 * dir.x * dir.z;
    sh.c[8] = 0.546274 * (dir.x * dir.x - dir.y * dir.y);
}

// ------------------------------------------------------------------

vec3 evaluate_sh9_irradiance(in vec3 direction)
{
    SH9 basis;

    project_onto_sh9(direction, basis);

    basis.c[0] *= CosineA0;
    basis.c[1] *= CosineA1;
    basis.c[2] *= CosineA1;
    basis.c[3] *= CosineA1;
    basis.c[4] *= CosineA2;
    basis.c[5] *= CosineA2;
    basis.c[6] *= CosineA2;
    basis.c[7] *= CosineA2;
    basis.c[8] *= CosineA2;

    vec3 color = vec3(0.0);

    for (int i = 0; i < 9; i++)
        color += texelFetch(s_SH, ivec2(i, 0), 0).rgb * basis.c[i];

    color.x = max(0.0, color.x);
    color.y = max(0.0, color.y);
    color.z = max(0.0, color.z);

    return color;
}

vec3 evaluate_sh9_irradiance(in SH9Color coef, in vec3 direction)
{
    SH9 basis;

    project_onto_sh9(direction, basis);

    basis.c[0] *= CosineA0;
    basis.c[1] *= CosineA1;
    basis.c[2] *= CosineA1;
    basis.c[3] *= CosineA1;
    basis.c[4] *= CosineA2;
    basis.c[5] *= CosineA2;
    basis.c[6] *= CosineA2;
    basis.c[7] *= CosineA2;
    basis.c[8] *= CosineA2;

    vec3 color = vec3(0.0);

    for (int i = 0; i < 9; i++)
        color += coef.c[i] * basis.c[i];

    color.x = max(0.0, color.x);
    color.y = max(0.0, color.y);
    color.z = max(0.0, color.z);

    return color;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    vec3 env_color;

    if (u_Type == 0) // Environment Map
        env_color = texture(s_Cubemap, FS_IN_WorldPos).rgb;
    else if (u_Type == 1) // Irradiance
    {
        env_color = evaluate_sh9_irradiance(normalize(FS_IN_WorldPos));

        env_color = env_color / Pi;
    }
    else if (u_Type == 2) // Prefilter
        env_color = textureLod(s_Prefilter, FS_IN_WorldPos, u_Roughness).rgb;
    else if (u_Type == 3) // Sky 
    {
        vec3 dir = normalize(FS_IN_WorldPos);

        float sun = step(cos(M_PI / 360.0), dot(dir, SUN_DIR));
                        
        vec3 sunColor = vec3(sun,sun,sun) * SUN_INTENSITY;

        vec3 extinction;
        vec3 inscatter = SkyRadiance(u_CameraPos, dir, extinction);
        vec3 col = sunColor * extinction + inscatter;

        env_color = col;
    }

    // HDR tonemap and gamma correct
    env_color = env_color / (env_color + vec3(1.0));
    env_color = pow(env_color, vec3(1.0 / 2.2));

    PS_OUT_Color = env_color;
}

// ------------------------------------------------------------------