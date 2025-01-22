#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real clearcoatGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1 - clearcoatGloss) * 0.1 + clearcoatGloss * 0.001;
    Real alpha_g2 = pow(alpha_g, 2);

    Vector3 half_vector = normalize(dir_in + dir_out);
    Vector3 h_l = to_local(frame, half_vector);

    Real eta = 1.5;
    Real R0 = pow(eta-1, 2) / pow(eta+1, 2);
    
    Real F_c = R0 + (1 - R0) * pow(1 - fabs(dot(half_vector, dir_out)), 5);
    Real D_c = (alpha_g2 - 1) / (c_PI * log(alpha_g2) * (1 + (alpha_g2 - 1) * pow(h_l.z, 2)));
    
    Vector3 w_l_in = to_local(frame, dir_in);
    Vector3 w_l_out = to_local(frame, dir_out);

    Real G_c = G_clearcoat(w_l_in) * G_clearcoat(w_l_out);
    
    return make_const_spectrum(F_c * D_c * G_c / (4 * fabs(dot(frame.n, dir_in))));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // call eval function to compute clearcoat_gloss
    Real clearcoatGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1 - clearcoatGloss) * 0.1 + clearcoatGloss * 0.001;

    Vector3 half_vector = normalize(dir_in + dir_out);
    Vector3 h_l = to_local(frame, half_vector);

    // compute 3 dot products - n.h n.out h.out
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_out = dot(half_vector, dir_out);

    // compute alpha^2
    Real alpha_g2 = pow(alpha_g, 2);
    Real D_c = (alpha_g2 - 1) / (c_PI * log(alpha_g2) * (1 + (alpha_g2 - 1) * pow(h_l.z, 2)));

    // if any of zero return 0 
    // calculate importance sample 
    if (n_dot_h <= 0 || n_dot_out <= 0 || h_dot_out <= 0) {
        return 0;
    }

    Real prob = D_c * fabs(dot(frame.n, half_vector)) / (4 * fabs(dot(half_vector, dir_out)));

    return prob;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // call eval function to compute clearcoat_gloss
    // compute alpha value
    Real clearcoatGloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1 - clearcoatGloss) * 0.1 + clearcoatGloss * 0.001;
    Real alpha_g2 = pow(alpha_g, 2);

    // equation 15
    // get local h
    Real h_elevation = acos(sqrt((1 - pow(alpha_g2, (1 - rnd_param_uv.x)))/(1-alpha_g2)));
    Real h_azimuth = 2 * c_PI * rnd_param_uv.y;
    
    Real h_x = sin(h_elevation) * cos(h_azimuth);
    Real h_y = sin(h_elevation) * sin(h_azimuth);
    Real h_z = cos(h_elevation);

    Vector3 h_l = Vector3{h_x, h_y, h_z};
    
    // transform micronormal to world space 
    Vector3 half_vector = to_world(frame, h_l);

    Real D_c = (alpha_g2 - 1) / (c_PI * log(alpha_g2) * (1 + (alpha_g2 - 1) * pow(h_l.z, 2)));
 
    // reflect over the world space normal 
    if (rnd_param_w < D_c) {
        // Sample from the specular lobe.
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        return BSDFSampleRecord{
            reflected,
            Real(0) /* eta */, Real(1) /* roughness */
        };
    } else {
        // Lambertian sampling
        return BSDFSampleRecord{
            to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
            Real(0) /* eta */, Real(1) /* roughness */};
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
