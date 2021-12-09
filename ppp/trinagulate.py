#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Sep 21 13:57:03 2021
"""

import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la


def main(x0=-1, xN=1, y0=-1, yN=1, xr=0, yr=0):
    # import meshpy.triangle as triangle
    import math

    points = [(x0, y0), (x0, yN), (xN, yN), (xN, y0)]

    def round_trip_connect(start, end):
        result = []
        for i in range(start, end):
            result.append((i, i + 1))
        result.append((end, start))

        return result

    def needs_refinement(vertices, area):
        vert_origin, vert_destination, vert_apex = vertices
        print(np.array(vert_origin), np.array(vert_destination), np.array(vert_apex))
        bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
        bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3
        # h1 = np.sqrt((vert_origin.x-vert_destination.x)**2 + (vert_origin.y-vert_destination.y)**2)
        # h2 = np.sqrt((vert_origin.x-vert_apex.x)**2 + (vert_origin.y-vert_apex.y)**2)
        # h3 = np.sqrt((vert_apex.x-vert_destination.x)**2 + (vert_apex.y-vert_destination.y)**2)
        # h

        dist_center = math.sqrt((bary_y - y0) ** 2)
        # print(dist_center)
        max_area = dist_center/5000 # math.fabs((dist_center - 0.1))
        print(area, max_area)
        # print(area, max_area)
        return

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    mesh = triangle.build(info, refinement_func=needs_refinement)

    return mesh

if __name__ == "__main__":
    mesh = main(0, .04, 0, .04, .02, 0)
    P = np.array(mesh.points)
    X, Y = P.T
    # print(mesh_points)
    T = np.array(mesh.elements)
    # print(np.array(mesh.elements))

    import matplotlib.pyplot as pt

    pt.triplot(X, Y, T, linewidth=1)
    pt.show()
