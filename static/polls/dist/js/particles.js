$(function () {
	var $container = $('<div id="particles"></div>');
	$container.insertBefore('.jumbotron .container');
	particlesJS(
		'particles',
		{
			particles: {
	            number: {
	                value: 12, // count of the circles
	                density: {
	                    enable: true,
	                    value_area: 1E3
	                }
	            },
	            color: {
	                value: "#e1e1e1" // color of the circles
	            },
	            shape: {
	                type: "circle",
	                stroke: {
	                    width: 0,
	                    color: "#000000"
	                },
	                polygon: {
	                    nb_sides: 5
	                }
	            },
	            opacity: {
	                value: 0.5,
	                random: false,
	                anim: {
	                    enable: false,
	                    speed: 1,
	                    opacity_min: 0.1,
	                    sync: false
	                }
	            },
	            size: {
	                value: 15,
	                random: false,
	                anim: {
	                    enable: false,
	                    speed: 180,
	                    size_min: 0.1,
	                    sync: false
	                }
	            },
	            line_linked: {
	                enable: true,
	                distance: 650,
	                color: "#cfcfcf", // color of the linked line
	                opacity: 0.26,
	                width: 1
	            },
	            move: {
	                enable: true,
	                speed: 2,
	                direction: "none",
	                random: true,
	                straight: false,
	                out_mode: "out",
	                bounce: false,
	                attract: {
	                    enable: false,
	                    rotateX: 600,
	                    rotateY: 1200
	                }
	            }
        	},
            interactivity: {
                detect_on: "canvas",
                events: {
                    onhover: {
                        enable: false,
                        mode: "repulse"
                    },
                    onclick: {
                        enable: false,
                        mode: "push"
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 400,
                        line_linked: {
                            opacity: 1
                        }
                    },
                    bubble: {
                        distance: 400,
                        size: 40,
                        duration: 2,
                        opacity: 8,
                        speed: 3
                    },
                    repulse: {
                        distance: 200,
                        duration: 0.4
                    },
                    push: {
                        particles_nb: 4
                    },
                    remove: {
                        particles_nb: 2
                    }
                }
            },
            retina_detect: true
        }
	);
});