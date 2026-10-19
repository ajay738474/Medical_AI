import React, { Suspense, useEffect, useRef } from "react";

import { Canvas, useThree } from "@react-three/fiber";

import {
    OrbitControls,
    Environment,
    Html,
    useGLTF,
} from "@react-three/drei";

import * as THREE from "three";


/* ============================================================
   LOADING
============================================================ */

function LoadingLungs() {
    return (
        <Html center>
            <div
                style={{
                    color: "white",
                    textAlign: "center",
                    fontFamily: "Arial, sans-serif",
                    background: "rgba(7, 16, 30, 0.94)",
                    padding: "20px 26px",
                    borderRadius: "14px",
                    minWidth: "210px",
                }}
            >
                <div
                    style={{
                        fontSize: "34px",
                        marginBottom: "8px",
                    }}
                >
                    🫁
                </div>

                <strong>
                    Loading Human Lungs...
                </strong>

                <div
                    style={{
                        marginTop: "7px",
                        fontSize: "11px",
                        color: "#aebbd0",
                    }}
                >
                    Loading anatomical 3D model
                </div>
            </div>
        </Html>
    );
}


/* ============================================================
   LUNG MODEL
============================================================ */

function LungModel() {
    const modelRef = useRef(null);

    const { scene } = useGLTF(
        "/models/lung/lung.glb"
    );


    useEffect(() => {

        if (!scene || !modelRef.current) {
            return;
        }


        console.log(
            "========================================"
        );

        console.log(
            "🫁 LUNG GLB LOADED"
        );


        /* ========================================================
           RESET
        ======================================================== */

        scene.position.set(
            0,
            0,
            0
        );

        scene.rotation.set(
            0,
            0,
            0
        );

        scene.scale.set(
            1,
            1,
            1
        );


        /* ========================================================
           PROCESS EVERY MESH
        ======================================================== */

        scene.traverse(
            (object) => {

                if (!object.isMesh) {
                    return;
                }


                object.visible = true;

                object.castShadow = true;

                object.receiveShadow = true;


                /* ====================================================
                   MATERIAL
                ==================================================== */

                if (!object.material) {
                    return;
                }


                const materials =
                    Array.isArray(object.material)
                        ? object.material
                        : [object.material];


                const newMaterials =
                    materials.map(
                        (oldMaterial) => {

                            const material =
                                oldMaterial.clone();


                            material.side =
                                THREE.DoubleSide;


                            material.transparent =
                                false;


                            material.opacity =
                                1;


                            material.depthWrite =
                                true;


                            material.depthTest =
                                true;


                            material.metalness =
                                0;


                            material.roughness =
                                0.48;


                            /*
                              If the model does not have
                              a texture, give it a natural
                              lung-like color.
                            */

                            if (!material.map) {

                                material.color =
                                    new THREE.Color(
                                        "#c96f82"
                                    );

                            }


                            /*
                              Slightly brighter
                              anatomical appearance.
                            */

                            if (
                                material.emissive
                            ) {

                                material.emissive =
                                    new THREE.Color(
                                        "#210a10"
                                    );

                                material.emissiveIntensity =
                                    0.08;

                            }


                            material.needsUpdate =
                                true;


                            return material;

                        }
                    );


                object.material =
                    Array.isArray(
                        object.material
                    )
                        ? newMaterials
                        : newMaterials[0];

            }
        );


        /* ========================================================
           FIND MODEL SIZE
        ======================================================== */

        scene.updateMatrixWorld(
            true
        );


        const box =
            new THREE.Box3().setFromObject(
                scene
            );


        const size =
            new THREE.Vector3();


        const center =
            new THREE.Vector3();


        box.getSize(
            size
        );


        box.getCenter(
            center
        );


        console.log(
            "🫁 Original size:",
            size
        );


        console.log(
            "🫁 Original center:",
            center
        );


        const maxDimension =
            Math.max(
                size.x,
                size.y,
                size.z
            );


        console.log(
            "🫁 Max dimension:",
            maxDimension
        );


        /* ========================================================
           SCALE
        ======================================================== */

        const targetSize = 4.2;


        let scale = 1;


        if (
            maxDimension > 0 &&
            Number.isFinite(
                maxDimension
            )
        ) {

            scale =
                targetSize /
                maxDimension;

        }


        console.log(
            "🫁 Calculated scale:",
            scale
        );


        modelRef.current.scale.setScalar(
            scale
        );


        /* ========================================================
           CENTER MODEL
        ======================================================== */

        modelRef.current.position.set(
            -center.x * scale,
            -center.y * scale,
            -center.z * scale
        );


        modelRef.current.rotation.set(
            0,
            0,
            0
        );


        modelRef.current.updateMatrixWorld(
            true
        );


        /* ========================================================
           FINAL SIZE
        ======================================================== */

        const finalBox =
            new THREE.Box3().setFromObject(
                modelRef.current
            );


        const finalSize =
            new THREE.Vector3();


        const finalCenter =
            new THREE.Vector3();


        finalBox.getSize(
            finalSize
        );


        finalBox.getCenter(
            finalCenter
        );


        console.log(
            "🫁 Final size:",
            finalSize
        );


        console.log(
            "🫁 Final center:",
            finalCenter
        );


        console.log(
            "========================================"
        );

        console.log(
            "🫁 LUNG MODEL READY"
        );

        console.log(
            "========================================"
        );

    }, [scene]);


    return (
        <group
            ref={modelRef}
        >

            <primitive
                object={scene}
            />

        </group>
    );
}


/* ============================================================
   LUNG SCENE
============================================================ */

function LungScene() {

    const {
        camera,
    } = useThree();


    /* ==========================================================
       CAMERA
    ========================================================== */

    useEffect(() => {

        camera.position.set(
            0,
            0,
            5.2
        );


        camera.lookAt(
            0,
            0,
            0
        );


        camera.near =
            0.01;


        camera.far =
            100;


        camera.updateProjectionMatrix();

    }, [camera]);


    return (
        <>

            {/* ====================================================
          LIGHTING
      ==================================================== */}

            <ambientLight
                intensity={1.5}
            />


            <hemisphereLight
                skyColor="#ffffff"
                groundColor="#34202a"
                intensity={1.8}
            />


            <directionalLight
                position={[
                    5,
                    7,
                    8,
                ]}
                intensity={3.5}
                castShadow
            />


            <directionalLight
                position={[
                    -5,
                    3,
                    5,
                ]}
                intensity={2.2}
            />


            <pointLight
                position={[
                    0,
                    1,
                    5,
                ]}
                intensity={1.5}
            />


            {/* ====================================================
          MODEL
      ==================================================== */}

            <Suspense
                fallback={
                    <LoadingLungs />
                }
            >

                <LungModel />

            </Suspense>


            {/* ====================================================
          ENVIRONMENT
      ==================================================== */}

            <Environment
                preset="studio"
            />


            {/* ====================================================
          CONTROLS
      ==================================================== */}

            <OrbitControls

                enableRotate

                enableZoom

                enablePan

                enableDamping

                dampingFactor={0.08}

                rotateSpeed={0.7}

                zoomSpeed={0.8}

                panSpeed={0.7}

                minDistance={1.5}

                maxDistance={10}

            />

        </>
    );
}


/* ============================================================
   MAIN COMPONENT
============================================================ */

export default function Lung3D() {

    return (

        <div
            className="lung-3d-container"
        >

            <Canvas

                camera={{
                    position: [
                        0,
                        0,
                        5.2,
                    ],

                    fov: 42,

                    near: 0.01,

                    far: 100,
                }}


                dpr={[
                    1,
                    2,
                ]}


                gl={{
                    antialias: true,

                    alpha: true,

                    powerPreference:
                        "high-performance",
                }}


                shadows

            >

                <LungScene />

            </Canvas>


            {/* ====================================================
          OVERLAY
      ==================================================== */}

            <div
                className="lung-3d-controls"
            >

                <div
                    className="lung-3d-title"
                >
                    🫁 Interactive Human Lungs
                </div>


                <div
                    className="lung-3d-help"
                >
                    Drag to rotate • Scroll to zoom •
                    Right-click to pan
                </div>


                <div
                    className="lung-3d-source"
                >
                    Anatomical reference model
                </div>

            </div>

        </div>
    );
}


/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/lung/lung.glb"
);