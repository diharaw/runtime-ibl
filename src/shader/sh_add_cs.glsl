// ------------------------------------------------------------------
// CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------

#define IMAGE_SIZE 16
#define CUBEMAP_SIZE 6

const float Pi = 3.141592654;

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout (local_size_x = 1, local_size_y = IMAGE_SIZE, local_size_z = CUBEMAP_SIZE) in;

// ------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------
// ------------------------------------------------------------------

layout (binding = 0, rgba32f) uniform image2D i_SH;

// ------------------------------------------------------------------
// SAMPLERS ---------------------------------------------------------
// ------------------------------------------------------------------

uniform sampler2DArray s_SHIntermediate;

// ------------------------------------------------------------------
// SHARED -----------------------------------------------------------
// ------------------------------------------------------------------

shared vec3 g_sh_coeffs[IMAGE_SIZE][CUBEMAP_SIZE];
shared float g_weights[IMAGE_SIZE][CUBEMAP_SIZE];

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    for (uint i = 0; i < (IMAGE_SIZE * CUBEMAP_SIZE); i++)
    {
        g_sh_coeffs[gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = vec3(0.0);
        g_weights[gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = 0.0;
    }

    barrier();

    // Add up coefficients along X axis.
    for (uint i = 0; i < IMAGE_SIZE; i++)
    {
        ivec3 p = ivec3(gl_GlobalInvocationID.x * IMAGE_SIZE + i, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
        vec4 val = texelFetch(s_SHIntermediate, p, 0);

        g_sh_coeffs[gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] += val.rgb;
        g_weights[gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] += val.a;
    }

    barrier();

    if (gl_GlobalInvocationID.z == 0)
    {
        // Add up coefficients along Z axis.
        for (uint i = 1; i < CUBEMAP_SIZE; i++)
        {
            g_sh_coeffs[gl_GlobalInvocationID.y][0] += g_sh_coeffs[gl_GlobalInvocationID.y][i];
            g_weights[gl_GlobalInvocationID.y][0] += g_weights[gl_GlobalInvocationID.y][i];
        }
    }

    barrier();

    if (gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0)
    {
        // Add up coefficients along Y axis.
        for (uint i = 1; i < IMAGE_SIZE; i++)
        {
            g_sh_coeffs[0][0] += g_sh_coeffs[i][0];
            g_weights[0][0] += g_weights[i][0];
        }

        float scale = (4.0 * Pi) / g_weights[0][0];

        // Write out the coefficents.
        imageStore(i_SH, ivec2(gl_GlobalInvocationID.x, 0), vec4(g_sh_coeffs[0][0] * scale, g_weights[0][0]));
    }
}

// ------------------------------------------------------------------