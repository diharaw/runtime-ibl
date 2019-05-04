const float Pi       = 3.141592654;
const float CosineA0 = Pi;
const float CosineA1 = (2.0 * Pi) / 3.0;
const float CosineA2 = Pi * 0.25;

out vec4 PS_OUT_Color;

in vec3 PS_IN_FragPos;
in vec3 PS_IN_Normal;
in vec2 PS_IN_TexCoord;

uniform sampler2D s_Albedo;
uniform sampler2D s_Normal;
uniform sampler2D s_Metallic;
uniform sampler2D s_Roughness;

uniform sampler2D s_BRDF;
uniform sampler2D s_IrradianceSH;
uniform samplerCube s_Prefiltered;

struct SH9
{
    float c[9];
};

// ------------------------------------------------------------------

struct SH9Color
{
    vec3 c[9];
};

vec3 normal_from_map()
{
    vec3 tangentNormal = texture(normalMap, PS_IN_TexCoord).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(PS_IN_FragPos);
    vec3 Q2  = dFdy(PS_IN_FragPos);
    vec2 st1 = dFdx(PS_IN_TexCoord);
    vec2 st2 = dFdy(PS_IN_TexCoord);

    vec3 N   = normalize(PS_IN_Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}
// ----------------------------------------------------------------------------
float distribution_ggx(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

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
        color += texelFetch(s_IrradianceSH, ivec2(i, 0), 0).rgb * basis.c[i];

    color.x = max(0.0, color.x);
    color.y = max(0.0, color.y);
    color.z = max(0.0, color.z);

    return color / Pi;
}

void main()
{
    // material properties
    vec3 albedo = texture(s_Albedo, PS_IN_TexCoord).rgb;
    float metallic = 1.0;//texture(s_Metallic, PS_IN_TexCoord).r;
    float roughness = 0.5;//texture(s_Roughness, PS_IN_TexCoord).r;

    // input lighting data
    vec3 N = normal_from_map();
    vec3 V = normalize(camPos - PS_IN_FragPos);
    vec3 R = reflect(-V, N); 

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    // reflectance equation
    vec3 Lo = vec3(0.0);
    // for(int i = 0; i < 4; ++i) 
    // {
    //     // calculate per-light radiance
    //     vec3 L = normalize(lightPositions[i] - PS_IN_FragPos);
    //     vec3 H = normalize(V + L);
    //     float distance = length(lightPositions[i] - PS_IN_FragPos);
    //     float attenuation = 1.0 / (distance * distance);
    //     vec3 radiance = lightColors[i] * attenuation;

    //     // Cook-Torrance BRDF
    //     float NDF = DistributionGGX(N, H, roughness);   
    //     float G   = GeometrySmith(N, V, L, roughness);    
    //     vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);        
        
    //     vec3 nominator    = NDF * G * F;
    //     float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; // 0.001 to prevent divide by zero.
    //     vec3 specular = nominator / denominator;
        
    //      // kS is equal to Fresnel
    //     vec3 kS = F;
    //     // for energy conservation, the diffuse and specular light can't
    //     // be above 1.0 (unless the surface emits light); to preserve this
    //     // relationship the diffuse component (kD) should equal 1.0 - kS.
    //     vec3 kD = vec3(1.0) - kS;
    //     // multiply kD by the inverse metalness such that only non-metals 
    //     // have diffuse lighting, or a linear blend if partly metal (pure metals
    //     // have no diffuse light).
    //     kD *= 1.0 - metallic;	                
            
    //     // scale light by NdotL
    //     float NdotL = max(dot(N, L), 0.0);        

    //     // add to outgoing radiance Lo
    //     Lo += (kD * albedo / PI + specular) * radiance * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    // }   
    
    // ambient lighting (we now use IBL as the ambient term)
    vec3 F = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;	  
    
    vec3 irradiance = evaluate_sh9_irradiance(N);
    vec3 diffuse      = irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(s_Prefiltered, R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(s_BRDF, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = (kD * diffuse + specular) * ao;
    
    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    PS_OUT_Color = vec4(color , 1.0);
}
