def render_row(data):
    '''
    Renders a row of width pixels, of the world of triangles.
    :param data: contains all the information needed to render a row
    '''

    has_warned_clip = False
    row = []
    for x in range(data["width"]):
        col = Vec3(0)

        for _ in range(SAMPLES_PER_PIXEL):
            xdir = (2 * (x + random()) * data["inv_width"] - 1) * data["angle"] * data["aspect_ratio"]
            ydir = (1 - 2 * (y + random()) * data["inv_height"]) * data["angle"]

            raydir = Vec3(xdir, ydir, 1)
            raydir.normalize()
            ray = Ray(data["camera_pos"], raydir)

            # Trace the ray with the bounding box
            col += trace_ray(ray, data["triangles"], data["lights"], data["bounding_box"])

        col /= SAMPLES_PER_PIXEL
        col = col.toList()
        clipped_col = clip(col, 0.0, 1.0)
        if not has_warned_clip:
            if col != clipped_col:
                warnings.warn("Image is clipping! Lights might be to bright..")
                has_warned_clip = True
        row.append(clipped_col)
    return row