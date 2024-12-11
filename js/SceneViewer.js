(function () {
    class SceneViewer {
        constructor(containerId, pointCloudPath, linesetPath, manager) {
            this.containerId = containerId;
            this.pointCloudPath = pointCloudPath;
            this.linesetPath = linesetPath;
            this.manager = manager;

            this.config = {
                camera: {
                    fov: 45,
                    near: 1,
                    far: 2000,
                    position: { x: 0, y: 0, z: 100 },
                },
                controls: {
                    minDistance: 36,
                    maxDistance: 180,
                },
                material: {
                    size: 1.25,
                    vertexColors: true,
                },
                scene: {
                    backgroundColor: 0xffffff,
                },
            };

            this.init();
        }

        getCanvasDimensions() {
            const canvasContainer = document.querySelector(".sv-canvas-container");
            const canvasWrapper = document.querySelector(".sv-canvas-wrapper");

            const wrapperStyle = window.getComputedStyle(canvasWrapper);
            const padding =
                parseFloat(wrapperStyle.paddingLeft) +
                parseFloat(wrapperStyle.paddingRight);
            const border =
                parseFloat(wrapperStyle.borderLeftWidth) +
                parseFloat(wrapperStyle.borderRightWidth);
            const gap = parseFloat(window.getComputedStyle(canvasContainer).gap);

            const wrapperPaddingBorder = padding + border;
            const totalPadding = wrapperPaddingBorder * 3 + gap * 2;

            const canvasWidth = (canvasContainer.clientWidth - totalPadding) / 3;
            const canvasHeight = canvasWidth * 2;

            return { canvasWidth, canvasHeight };
        }

        createCamera(width, height) {
            const camera = new THREE.PerspectiveCamera(
                this.config.camera.fov,
                width / height,
                this.config.camera.near,
                this.config.camera.far
            );
            camera.position.set(
                this.config.camera.position.x,
                this.config.camera.position.y,
                this.config.camera.position.z
            );
            return camera;
        }

        createScene() {
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(this.config.scene.backgroundColor);
            return scene;
        }

        createRenderer(width, height) {
            const renderer = new THREE.WebGLRenderer({ antialias: true }); // Enable antialiasing
            renderer.setSize(width, height);
            document.getElementById(this.containerId).appendChild(renderer.domElement);
            return renderer;
        }

        loadPointCloud() {
            const loader = new THREE.PLYLoader();
            loader.load(this.pointCloudPath, (geometry) => {
                const material = this.createCustomPointMaterial();
                const points = new THREE.Points(geometry, material);
                this.scene.add(points);
                this.loadLineset(); // Load lineset after point cloud
                this.render();
            });
        }

        loadLineset() {
            const loader = new THREE.PLYLoader();
            loader.load(this.linesetPath, (geometry) => {
                const material = new THREE.LineBasicMaterial({ vertexColors: true });
                const lineset = new THREE.LineSegments(geometry, material);
                this.scene.add(lineset);
                this.render();
            });
        }

        createCustomPointMaterial() {
            return new THREE.ShaderMaterial({
                uniforms: {
                    pointSize: { value: this.config.material.size },
                },
                vertexShader: `
            uniform float pointSize;
            varying vec3 vColor;
            void main() {
              vColor = color;
              gl_PointSize = pointSize;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
                fragmentShader: `
            varying vec3 vColor;
            void main() {
              gl_FragColor = vec4(vColor, 1.0);
            }
          `,
                vertexColors: true,
            });
        }

        createControls(camera, domElement) {
            const controls = new THREE.OrbitControls(camera, domElement);
            controls.minDistance = this.config.controls.minDistance;
            controls.maxDistance = this.config.controls.maxDistance;
            return controls;
        }

        addAxesHelper() {
            const axesHelper = new THREE.AxesHelper(5);
            axesHelper.rotation.z = Math.PI / 2;
            this.scene.add(axesHelper);
        }

        init() {
            const { canvasWidth, canvasHeight } = this.getCanvasDimensions();

            this.camera = this.createCamera(canvasWidth, canvasHeight);
            this.scene = this.createScene();
            this.renderer = this.createRenderer(canvasWidth, canvasHeight);

            const cursor = document.createElement('div');
            cursor.className = 'sv-interactive-cursor';
            document.getElementById(this.containerId).parentNode.appendChild(cursor);
            this.cursor = cursor;

            this.renderer.domElement.addEventListener('pointerdown', () => {
                this.manager.hideAllCursors();
            });

            this.addAxesHelper();
            this.loadPointCloud();

            this.controls = this.createControls(this.camera, this.renderer.domElement);
            this.controls.addEventListener("change", () =>
                this.manager.synchronizeControls(this)
            );

            this.attachControlButtonListeners();
        }

        attachControlButtonListeners() {
            const wrapper = document.getElementById(this.containerId).parentNode;
            const zoomInButton = wrapper.querySelector(".sv-zoom-in-button");
            const zoomOutButton = wrapper.querySelector(".sv-zoom-out-button");
            const resetButton = wrapper.querySelector(".sv-reset-button");

            zoomInButton.addEventListener("click", () => this.simulateZoom(true));
            zoomOutButton.addEventListener("click", () => this.simulateZoom(false));
            resetButton.addEventListener("click", () => this.resetView());
        }

        simulateZoom(isZoomIn) {
            const event = new WheelEvent("wheel", {
                deltaY: isZoomIn ? -100 : 100,
                bubbles: true,
                cancelable: true,
            });
            this.renderer.domElement.dispatchEvent(event);
        }

        adjustCanvasSize() {
            const { canvasWidth, canvasHeight } = this.getCanvasDimensions();

            this.camera.aspect = canvasWidth / canvasHeight;
            this.camera.updateProjectionMatrix();

            this.renderer.setSize(canvasWidth, canvasHeight);
        }

        render() {
            this.renderer.render(this.scene, this.camera);
        }

        resetView() {
            this.camera.position.set(
                this.config.camera.position.x,
                this.config.camera.position.y,
                this.config.camera.position.z
            );
            this.controls.target.set(0, 0, 0);
            this.controls.update();
            this.render();
        }
    }

    class ViewerManager {
        constructor() {
            this.viewers = [];
            this.syncing = false;
            this.cursorsHidden = false;
        }

        addViewer(viewer) {
            this.viewers.push(viewer);
        }

        synchronizeControls(activeViewer) {
            if (this.syncing) return;
            this.syncing = true;

            const { camera, controls } = activeViewer;

            this.viewers.forEach((viewer) => {
                if (viewer !== activeViewer) {
                    viewer.camera.position.copy(camera.position);
                    viewer.camera.rotation.copy(camera.rotation);
                    viewer.controls.target.copy(controls.target);
                    viewer.controls.update();
                }
                viewer.render();
            });

            this.syncing = false;
        }

        adjustCanvasSizes() {
            this.viewers.forEach((viewer) => viewer.adjustCanvasSize());
        }

        renderAll() {
            this.viewers.forEach((viewer) => viewer.render());
        }

        getCanvasDimensions() {
            const canvasContainer = document.querySelector(".sv-canvas-container");
            const canvasWrapper = document.querySelector(".sv-canvas-wrapper");

            const wrapperStyle = window.getComputedStyle(canvasWrapper);
            const padding =
                parseFloat(wrapperStyle.paddingLeft) +
                parseFloat(wrapperStyle.paddingRight);
            const border =
                parseFloat(wrapperStyle.borderLeftWidth) +
                parseFloat(wrapperStyle.borderRightWidth);
            const gap = parseFloat(window.getComputedStyle(canvasContainer).gap);

            const wrapperPaddingBorder = padding + border;
            const totalPadding = wrapperPaddingBorder * this.viewers.length + gap * (this.viewers.length - 1);

            const canvasWidth = (canvasContainer.clientWidth - totalPadding) / this.viewers.length;
            const canvasHeight = canvasWidth * 2;

            return { canvasWidth, canvasHeight };
        }

        hideAllCursors() {
            if (this.cursorsHidden) return;
            this.cursorsHidden = true;

            this.viewers.forEach(viewer => {
                if (viewer.cursor) {
                    viewer.cursor.classList.add('hidden');
                }
            });
        }
    }

    function init() {
        const manager = new ViewerManager();

        const waymoViewer = new SceneViewer(
            "canvas-waymo",
            "assets/pcd_waymo.ply",
            "assets/lineset.ply",
            manager
        );
        const nuscenesViewer = new SceneViewer(
            "canvas-nuscenes",
            "assets/pcd_nuscenes.ply",
            "assets/lineset.ply",
            manager
        );
        const kittiViewer = new SceneViewer(
            "canvas-kitti",
            "assets/pcd_kitti.ply",
            "assets/lineset.ply",
            manager
        );

        manager.addViewer(waymoViewer);
        manager.addViewer(nuscenesViewer);
        manager.addViewer(kittiViewer);

        window.addEventListener("resize", () => {
            manager.adjustCanvasSizes();
            manager.renderAll();
        });

        animate(manager);
    }

    function animate(manager) {
        requestAnimationFrame(() => animate(manager));
        manager.viewers.forEach((viewer) => viewer.controls.update());
    }

    document.addEventListener("DOMContentLoaded", init);
})();
