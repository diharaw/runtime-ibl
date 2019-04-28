#define SH_DEBUG

out vec3 PS_OUT_Color;

in vec3 FS_IN_WorldPos;

uniform samplerCube s_Cubemap;
uniform sampler2D s_SH;

uniform int u_Type;

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
    else if (u_Type == 2) //Debug irradiance
    {
        SH9Color coef;

        coef.c[0] = vec3(1.587685, 0.938743, 0.946143);
        coef.c[1] = vec3(0.964110, 0.260693, -0.252178);
        coef.c[2] = vec3(-0.284860, 0.083329, 0.312422);
        coef.c[3] = vec3(-0.268414, -0.068907, 0.142706);
        coef.c[4] = vec3(0.014640, -0.082044, -0.186975);
        coef.c[5] = vec3(-0.233512, -0.225626, -0.324495);
        coef.c[6] = vec3(0.147594, 0.146880, 0.191309);
        coef.c[7] = vec3(-0.077331, -0.163117, -0.227645);
        coef.c[8] = vec3(-0.270700, -0.070550, 0.029554);

        env_color = evaluate_sh9_irradiance(coef, normalize(FS_IN_WorldPos));

        env_color = env_color / Pi;
    }

    // HDR tonemap and gamma correct
    env_color = env_color / (env_color + vec3(1.0));
    env_color = pow(env_color, vec3(1.0 / 2.2));

    PS_OUT_Color = env_color;
}

// ------------------------------------------------------------------