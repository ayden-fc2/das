"use client"

import {Box} from "@mui/material";
import SvgRender from "@/app/components/draw/ComponentRender";
import React from "react";
import ComponentRender from "@/app/components/draw/ComponentRender";

const RelayCom = ()=> {
    const demo = {
        "LINE": [
            {
                "plotstyle_flags": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "thickness": 0,
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 19,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    424,
                    733
                ],
                "z_is_zero": 1,
                "has_ds_data": 0,
                "linewt": 29,
                "extrusion": [
                    0,
                    0,
                    1
                ],
                "end": [
                    23.10626043153752,
                    27.62223606800217,
                    0
                ],
                "has_face_visualstyle": 0,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bitsize": 306,
                "start": [
                    77.02708294867625,
                    27.62223606800217,
                    0
                ],
                "index": 179,
                "handle": [
                    0,
                    2,
                    1157
                ],
                "entmode": 0,
                "_subclass": "AcDbLine",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 44,
                "has_edge_visualstyle": 0,
                "entity": "LINE"
            }
        ],
        "CIRCLE": [
            {
                "plotstyle_flags": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "thickness": 0,
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 18,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    423,
                    733
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "extrusion": [
                    0,
                    0,
                    1
                ],
                "has_face_visualstyle": 0,
                "radius": 36.21107107755127,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bitsize": 277,
                "center": [
                    24.74021930815434,
                    26.15272170066419,
                    0
                ],
                "index": 178,
                "handle": [
                    0,
                    2,
                    1156
                ],
                "entmode": 0,
                "_subclass": "AcDbCircle",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 40,
                "has_edge_visualstyle": 0,
                "entity": "CIRCLE"
            }
        ],
        "POINT": [],
        "LWPOLYLINE": [
            {
                "plotstyle_flags": 0,
                "flag": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 77,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    425,
                    733
                ],
                "points": [
                    [
                        26.7009998915637,
                        76.9328138605797
                    ],
                    [
                        14.60964434166817,
                        51.29784211940751
                    ],
                    [
                        23.10626043153752,
                        27.62223606800217
                    ]
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "has_face_visualstyle": 0,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bulges": [],
                "bitsize": 479,
                "index": 180,
                "handle": [
                    0,
                    2,
                    1158
                ],
                "entmode": 0,
                "_subclass": "AcDbPolyline",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 65,
                "has_edge_visualstyle": 0,
                "vertexids": [],
                "entity": "LWPOLYLINE"
            },
            {
                "plotstyle_flags": 0,
                "flag": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 77,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    427,
                    733
                ],
                "points": [
                    [
                        -17.67622972864251,
                        80.55117291347551
                    ],
                    [
                        -22.00589229760169,
                        58.37745694956544
                    ],
                    [
                        -4.86764295917521,
                        58.37745694956544
                    ],
                    [
                        -17.67622972864251,
                        80.55117291347551
                    ]
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "has_face_visualstyle": 0,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bulges": [],
                "bitsize": 547,
                "index": 182,
                "handle": [
                    0,
                    2,
                    1160
                ],
                "entmode": 0,
                "_subclass": "AcDbPolyline",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 74,
                "has_edge_visualstyle": 0,
                "vertexids": [],
                "entity": "LWPOLYLINE"
            }
        ],
        "MTEXT": [],
        "ARC": [
            {
                "plotstyle_flags": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "thickness": 0,
                "start_angle": 5.50689155015097,
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 17,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    426,
                    733
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "extrusion": [
                    0,
                    0,
                    1
                ],
                "has_face_visualstyle": 0,
                "radius": 38.10274175096269,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bitsize": 409,
                "center": [
                    -2.44666509516844,
                    -0.54357477245957,
                    0
                ],
                "index": 181,
                "handle": [
                    0,
                    2,
                    1159
                ],
                "entmode": 0,
                "_subclass": "AcDbArc",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 57,
                "has_edge_visualstyle": 0,
                "end_angle": 2.05530149116747,
                "entity": "ARC"
            },
            {
                "plotstyle_flags": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "thickness": 0,
                "start_angle": 1.88553591965387,
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 17,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    428,
                    733
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "extrusion": [
                    0,
                    0,
                    1
                ],
                "has_face_visualstyle": 0,
                "radius": 8.26832638785986,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bitsize": 409,
                "center": [
                    -47.94989085380504,
                    19.86869541943607,
                    0
                ],
                "index": 183,
                "handle": [
                    0,
                    2,
                    1161
                ],
                "entmode": 0,
                "_subclass": "AcDbArc",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 57,
                "has_edge_visualstyle": 0,
                "end_angle": 6.25292386083797,
                "entity": "ARC"
            },
            {
                "plotstyle_flags": 0,
                "color": {
                    "flag": 1,
                    "index": 256,
                    "rgb": "000000"
                },
                "thickness": 0,
                "start_angle": 3.00532478277947,
                "has_full_visualstyle": 0,
                "invisible": 0,
                "type": 17,
                "layer": [
                    5,
                    1,
                    16,
                    16
                ],
                "ltype_flags": 0,
                "ownerhandle": [
                    12,
                    2,
                    429,
                    733
                ],
                "has_ds_data": 0,
                "linewt": 29,
                "extrusion": [
                    0,
                    0,
                    1
                ],
                "has_face_visualstyle": 0,
                "radius": 7.23212140175825,
                "is_xdic_missing": 1,
                "material_flags": 0,
                "ltype_scale": 1,
                "bitsize": 409,
                "center": [
                    -20.61369793840134,
                    3.49303765607014,
                    0
                ],
                "index": 184,
                "handle": [
                    0,
                    2,
                    1162
                ],
                "entmode": 0,
                "_subclass": "AcDbArc",
                "preview_exists": 0,
                "shadow_flags": 0,
                "size": 57,
                "has_edge_visualstyle": 0,
                "end_angle": 1.38811398326473,
                "entity": "ARC"
            }
        ],
        "TEXT": [],
        "INSERTS": []
    }
    return (
        <Box className={`w-full h-full text-black overscroll-y-auto overflow-x-hidden`}>
            依赖关系分析TODO
            <ComponentRender {...demo} />
        </Box>
    )
}

export default RelayCom