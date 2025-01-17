Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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
    Vector3 half_vector = normalize(dir_in + dir_out);
    
    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    auto F = [&](Real F_90, const Vector3 &n, const Vector3 &w) -> Real {
        return (1 + (F_90 - 1)) * (1 - pow(fabs(dot(n, w)), 5));
    };

    Real F_D90 = 0.5 + Real(2) * roughness * pow(fabs(dot(half_vector, dir_out)), 2);
    Real F_D_in = F(F_D90, frame.n, dir_in);
    Real F_D_out = F(F_D90, frame.n, dir_out);

    Spectrum f_base_diffuse = baseColor / c_PI * F_D_in * F_D_out * fabs(dot(frame.n, dir_out));

    Real F_SS90 = roughness * pow(fabs(dot(half_vector, dir_out)), 2);
    Real F_SS_in = F(F_SS90, frame.n, dir_in);
    Real F_SS_out = F(F_SS90, frame.n, dir_out);

    Spectrum f_subsurface = 1.25 * baseColor / c_PI * 
                            (F_SS_in * F_SS_out * (1 / (dot(frame.n, dir_in) + dot(frame.n, dir_out)) - 0.5) + 0.5) 
                            * fabs(dot(frame.n, dir_out));

    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);

    return (1 - subsurface) * f_base_diffuse + subsurface * f_subsurface;

}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    // Homework 1: implement this!
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, roughness};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
