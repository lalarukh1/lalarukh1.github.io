    //Main slides
    $(function () {
        let controller = new ScrollMagic.Controller();
        let wipeAnimation = new TimelineMax()
            .fromTo("section.panel.second", 1, {x: "100%"}, {x: "0%", ease: Linear.easeNone})  // in from left
            .fromTo("section.panel.third",    1, {x:  "-120%"}, {x: "0%", ease: Linear.easeNone})  // in from right
            .fromTo("section.panel.fourth", 1, {y: "-100%"}, {y: "0%", ease: Linear.easeNone}); // in from top
        new ScrollMagic.Scene({
            triggerElement: "#pinContainer",
            triggerHook: "onLeave",
            duration: "300%"
        })
            .setPin("#pinContainer")
            .setTween(wipeAnimation)
            .addTo(controller);

        //rotating mobile
        let controller_h = new ScrollMagic.Controller();
        let scene = new ScrollMagic.Scene({
            offset: 200,
            duration: "400"
        })
            .addTo(controller_h)
            .setTween(TweenMax.to("#animate", 4, { rotation:-90, transformOrigin:"20em 10em"})) ;


        //replacing photos
        var controller_2 = new ScrollMagic.Controller({});
        var tween = new TimelineMax ()
            .add([
                TweenMax.to("#parallaxContainer .layer3", 1, {opacity: 1, x: "-121%"}, { ease: Linear.easeNone}),
                TweenMax.to('.iphone-content', 1, {scale: -0.1 })
            ]);

        var scene_2 = new ScrollMagic.Scene({
            duration: "430",
            offset: 400
        })
            .setTween(tween)
            .addTo(controller_2)
    });

