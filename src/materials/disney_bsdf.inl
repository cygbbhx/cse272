#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Real specularTransmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specularTint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);    
    
    Spectrum f_diffuse = eval(DisneyDiffuse{bsdf.base_color, bsdf.roughness, bsdf.subsurface}, dir_in, dir_out, vertex, texture_pool, dir);
    Spectrum f_glass = eval(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta}, dir_in, dir_out, vertex, texture_pool, dir);
    Spectrum f_clearcoat = eval(DisneyClearcoat{bsdf.clearcoat_gloss}, dir_in, dir_out, vertex, texture_pool, dir);
    Spectrum f_sheen = eval(DisneySheen{bsdf.base_color, bsdf.sheen_tint}, dir_in, dir_out, vertex, texture_pool, dir);

    //// f_metal_hat
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    Vector3 h_l = to_local(frame, half_vector);
    Vector3 w_l_in = to_local(frame, dir_in);
    Vector3 w_l_out = to_local(frame, dir_out);

    Spectrum tint = luminance(baseColor) > 0 ? baseColor / luminance(baseColor) : make_const_spectrum(1);

    Spectrum K_s = (1 - specularTint) + specularTint * tint;
    Spectrum C_0 = specular * (pow(eta-1, 2) / pow(eta+1, 2)) * (1 - metallic) * K_s + metallic * baseColor;
    Spectrum F_m_hat = C_0 + (1 - C_0) * pow(1 - dot(half_vector, dir_out), 5);

    Real D_m = D_metal(h_l, roughness, anisotropic);
    Real G_m = G_metal(w_l_in, roughness, anisotropic) * G_metal(w_l_out, roughness, anisotropic);
    Spectrum f_metal_hat = (F_m_hat * D_m * G_m) / (4 * fabs(dot(frame.n, dir_in)));
    ////
    
    if (dot(dir_in, vertex.geometric_normal) <= 0) {
        f_diffuse = make_zero_spectrum();
        f_metal_hat = make_zero_spectrum();
        f_clearcoat = make_zero_spectrum();
        f_sheen = make_zero_spectrum();
    }
    
    return (1 - specularTransmission) * (1 - metallic) * f_diffuse +
            (1 - metallic) * sheen * f_sheen +
            (1 - specularTransmission * (1 - metallic)) * f_metal_hat +
            0.25 * clearcoat * f_clearcoat +
            (1 - metallic) * specularTransmission * f_glass;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real specularTransmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);    

    Real diffuseWeight = (1 - metallic) * (1 - specularTransmission);
    Real metalWeight = (1 - specularTransmission) * (1 - metallic);
    Real glassWeight = (1 - metallic) * specularTransmission;
    Real clearcoatWeight = 0.25 * clearcoat;

    // choose diffuse/metal/clearcoat/glass randomly based on the weights
    Real totalWeight = diffuseWeight + metalWeight + clearcoatWeight + glassWeight;
    diffuseWeight /= totalWeight;
    metalWeight /= totalWeight;
    clearcoatWeight /= totalWeight;
    glassWeight /= totalWeight;

    Real diffuse_prob = pdf_sample_bsdf(DisneyDiffuse{bsdf.base_color, bsdf.roughness, bsdf.subsurface}, dir_in, dir_out, vertex, texture_pool, dir);

    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }

    Vector3 h_l = to_local(frame, half_vector);
    Vector3 w_l_in = to_local(frame, dir_in);
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real D_m = D_metal(h_l, roughness, anisotropic);
    Real G_in = G_metal(w_l_in, roughness, anisotropic);

    Real metal_prob = D_m * G_in / (4.0 * fabs(n_dot_in));

    // Real metal_prob = pdf_sample_bsdf(DisneyMetal{bsdf.base_color, bsdf.roughness, bsdf.anisotropic}, dir_in, dir_out, vertex, texture_pool, dir);
    Real glass_prob = pdf_sample_bsdf(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta}, dir_in, dir_out, vertex, texture_pool, dir);
    Real clearcoat_prob = pdf_sample_bsdf(DisneyClearcoat{bsdf.clearcoat_gloss}, dir_in, dir_out, vertex, texture_pool, dir);

    if (dot(dir_in, vertex.geometric_normal) <= 0) {
        return glass_prob;
    } else {
        return diffuse_prob * diffuseWeight + glass_prob * glassWeight + metal_prob * metalWeight + clearcoat_prob * clearcoatWeight;
    }

}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real specularTransmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);    

    Real diffuseWeight = (1 - metallic) * (1 - specularTransmission);
    Real metalWeight = (1 - specularTransmission) * (1 - metallic);
    Real glassWeight = (1 - metallic) * specularTransmission;
    Real clearcoatWeight = 0.25 * clearcoat;

    // choose diffuse/metal/clearcoat/glass randomly based on the weights
    Real totalWeight = diffuseWeight + metalWeight + clearcoatWeight + glassWeight;
    diffuseWeight /= totalWeight;
    metalWeight /= totalWeight;
    clearcoatWeight /= totalWeight;
    glassWeight /= totalWeight;

    if (dot(dir_in, vertex.geometric_normal) <= 0) {
        return sample_bsdf(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic}, dir_in, vertex, texture_pool, rnd_param_uv, rnd_param_w, dir);
    }

    Real metal_prob = diffuseWeight + metalWeight;
    Real clearcoat_prob = metal_prob + clearcoatWeight;
    Real glass_prob = clearcoat_prob + glassWeight;
    Real scaled_w = 0;

    if (rnd_param_w <= diffuseWeight) {
        scaled_w = rnd_param_w / diffuseWeight;
        return sample_bsdf(DisneyDiffuse{bsdf.base_color, bsdf.roughness, bsdf.subsurface}, dir_in,vertex, texture_pool, rnd_param_uv, scaled_w, dir);
    } else if (rnd_param_w <= metal_prob) {
        scaled_w = (rnd_param_w - diffuseWeight) / (metal_prob - diffuseWeight);
        return sample_bsdf(DisneyMetal{bsdf.base_color, bsdf.roughness, bsdf.anisotropic}, dir_in, vertex, texture_pool, rnd_param_uv, scaled_w, dir);
    } else if (rnd_param_w <= clearcoat_prob) {
        scaled_w = (rnd_param_w - metal_prob) / (clearcoat_prob - metal_prob);
        return sample_bsdf(DisneyClearcoat{bsdf.clearcoat_gloss}, dir_in, vertex, texture_pool, rnd_param_uv, scaled_w, dir);
    } else {
        Real prob = 1 - glassWeight;
        scaled_w = (rnd_param_w - prob) / (1 - prob);
        return sample_bsdf(DisneyGlass{bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta}, dir_in, vertex, texture_pool, rnd_param_uv, scaled_w, dir);
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
