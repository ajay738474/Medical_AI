import React, { useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

/* ============================================================
   EAR MODEL
============================================================ */

function EarModel() {

    const { scene } = useGLTF(
        "/models/ear/ear.glb"
    );

    /* ==========================================================
       PROCESS MODEL
    ========================================================== */

    useEffect(() => {

        scene.updateMatrixWorld(true);

        const meshes = [];

        scene.traverse((child) => {

            if (!child.isMesh) {
                return;
            }

            meshes.push(child);

            child.castShadow = true;
            child.receiveShadow = true;

            if (child.material) {

                const materials =
                    Array.isArray(child.material)
                        ? child.material
                        : [child.material];

                materials.forEach((material) => {

                    material.side =
                        THREE.DoubleSide;

                    material.roughness = 0.5;
                    material.metalness = 0;

                    material.needsUpdate = true;

                });
            }

        });

        /* ========================================================
           HIDE BACKING BOARD
        ======================================================== */

        meshes.forEach((mesh) => {

            const name =
                String(
                    mesh.name || ""
                ).toLowerCase();

            const parentName =
                String(
                    mesh.parent?.name || ""
                ).toLowerCase();

            const combinedName =
                `${name} ${parentName}`;

            /* ------------------------------------------------------
               Hide obvious board/backplate meshes
            ------------------------------------------------------ */

            if (
                combinedName.includes("board") ||
                combinedName.includes("backboard") ||
                combinedName.includes("backing") ||
                combinedName.includes("backplate") ||
                combinedName.includes("plate") ||
                combinedName.includes("base")
            ) {

                mesh.visible = false;

                return;
            }

        });

        /* ========================================================
           DETECT LARGE FLAT BACKGROUND MESH
        ======================================================== */

        const visibleMeshes =
            meshes.filter(
                (mesh) => mesh.visible
            );

        let largestFlatMesh = null;
        let largestArea = 0;

        visibleMeshes.forEach((mesh) => {

            const box =
                new THREE.Box3().setFromObject(
                    mesh
                );

            const size =
                new THREE.Vector3();

            box.getSize(size);

            const dimensions = [
                size.x,
                size.y,
                size.z,
            ].sort(
                (a, b) => b - a
            );

            const largest =
                dimensions[0];

            const middle =
                dimensions[1];

            const smallest =
                dimensions[2];

            if (
                largest <= 0 ||
                middle <= 0 ||
                smallest <= 0
            ) {
                return;
            }

            /* ------------------------------------------------------
               A backing board is usually:
               - very large
               - very flat
               - one dimension extremely thin
            ------------------------------------------------------ */

            const flatRatio =
                smallest / largest;

            const surfaceArea =
                largest * middle;

            if (
                flatRatio < 0.025 &&
                surfaceArea > largestArea
            ) {

                largestArea =
                    surfaceArea;

                largestFlatMesh =
                    mesh;
            }

        });

        /* ========================================================
           HIDE THE LARGEST FLAT BOARD
        ======================================================== */

        if (largestFlatMesh) {

            largestFlatMesh.visible =
                false;

            console.log(
                "EAR 3D: Backing board hidden:",
                largestFlatMesh.name
            );

        }

        console.log(
            "EAR 3D: Meshes found:",
            meshes.length
        );

    }, [scene]);

    return (
        <primitive
            object={scene}
            scale={1}
        />
    );
}

/* ============================================================
   EAR 3D COMPONENT
============================================================ */

export default function Ear3D() {

    return (
        <div className="ear-3d-container">

            {/* ======================================================
          CANVAS
      ====================================================== */}

            <Canvas
                camera={{
                    position: [0, 0, 4.5],
                    fov: 38,
                    near: 0.01,
                    far: 100,
                }}

                dpr={[1, 2]}

                shadows

                gl={{
                    antialias: true,
                    alpha: false,
                    powerPreference:
                        "high-performance",
                }}
            >

                {/* ====================================================
            BACKGROUND
        ==================================================== */}

                <color
                    attach="background"
                    args={["#101522"]}
                />

                {/* ====================================================
            ENVIRONMENT
        ==================================================== */}

                <Environment
                    preset="studio"
                />

                {/* ====================================================
            LIGHTING
        ==================================================== */}

                <ambientLight
                    intensity={1.5}
                />

                <hemisphereLight
                    intensity={1.5}
                    groundColor="#111827"
                    skyColor="#ffffff"
                />

                <directionalLight
                    position={[5, 6, 7]}
                    intensity={3}
                    castShadow
                />

                <directionalLight
                    position={[-5, 3, 5]}
                    intensity={2}
                />

                <directionalLight
                    position={[0, -3, 4]}
                    intensity={1.5}
                />

                <pointLight
                    position={[0, 2, 3]}
                    intensity={1.5}
                />

                {/* ====================================================
            EAR
        ==================================================== */}

                <EarModel />

                {/* ====================================================
            CONTROLS
        ==================================================== */}

                <OrbitControls

                    enableRotate={true}

                    enableZoom={true}

                    enablePan={true}

                    rotateSpeed={0.8}

                    zoomSpeed={0.8}

                    panSpeed={0.8}

                    enableDamping={true}

                    dampingFactor={0.08}

                    minDistance={1}

                    maxDistance={10}

                    target={[0, 0, 0]}

                />

            </Canvas>

            {/* ======================================================
          TITLE
      ====================================================== */}

            <div className="ear-3d-title">

                <span>
                    👂
                </span>

                <span>
                    Interactive Human Ear
                </span>

            </div>

            {/* ======================================================
          HELP
      ====================================================== */}

            <div className="ear-3d-help">

                Drag to rotate • Scroll to zoom • Right-click to pan

            </div>

        </div>
    );
}

/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/ear/ear.glb"
);