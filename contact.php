<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Portfolio</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.1/css/bulma.min.css">
    <script src="https://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>
    <script defer src="https://use.fontawesome.com/releases/v5.3.1/js/all.js"></script>
    <link rel="stylesheet" href="bulma-carousel.min.css">
    <title> My Blog </title>
    <link href="https://fonts.googleapis.com/css?family=Alegreya+SC|Raleway|Great+Vibes" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=Merriweather" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.0/animate.min.css">

    <link href="style.css" rel="stylesheet" type="text/css" media="all">
</head>
<body>
<section class="section mainSection">
    <nav class="navbar" role="navigation" aria-label="main navigation">
        <span class="nav"><span class="icon"><i class="fas fa-home fa-xs"></i></span></a><span>Home</span></span>
        <span class="nav"><span class="icon"><i class="fas fa-user fa-xs"></i></span><span>About Me</span></span>
        <span class="nav"><span class="icon"><i class="fas fa-portrait fa-xs"></i></span><span>Portfolio</span></span>
        <span class="nav"><span class="icon"><i class="fas fa-file fa-xs"></i></span><span>Resume</span></span>
        <span class="nav"><span class="icon"><i
                class="fas fa-file-signature fa-xs"></i></span><span>Contact</span></span>
    </nav>

    <div class="columns">
        <div class="column is-narrow-widescreen is-narrow-desktop is-narrow-tablet is-3-mobile sideBar">
            <p class="buttons is-centered">
                <a class="button">
                        <span class="icon">
                        <i class="fab fa-facebook"></i>
                        </span>
                </a>
                <a class="button">
                        <span class="icon">
                        <i class="fab fa-github"></i>
                        </span>
                </a>
                <a class="button">
                        <span class="icon">
                        <i class="fab fa-linkedin-in"></i>
                        </span>
                </a>
            </p>

            <figure class="image is-square">
                <img src="https://avatars0.githubusercontent.com/u/35193027?s=460&v=4" alt="Image">
            </figure>

            <div class="column nav contactNav is-marginless is-paddingless">
                <p class="homeNav"><span class="icon homeI"><i class="fas fa-home fa-xs"></i></span><a href="file:///C:/wamp2/www/bulma/index.html"><span>Home</span></a></p>
                <p class="aboutNav"><span class="icon aboutI"><i class="fas fa-user fa-xs"></i></span><a href="file:///C:/wamp2/www/bulma/about-me.html"><span>About Me</span></a></p>
                <p class="portfolioNav"><span class="icon portfolioI"><i class="fas fa-portrait fa-xs"></i></span><a href="file:///C:/wamp2/www/bulma/portfolio.html"><span>Portfolio</span></a></p>
                <p class="resumeNav"><span class="icon resumeI"><i class="fas fa-file fa-xs"></i></span><a href="file:///C:/wamp2/www/bulma/resume.html"><span>Resume</span></a></p>
                <p class="contactNav is-active"><span class="icon contactI"><i class="fas fa-file-signature fa-xs"></i></span><a href="file:///C:/wamp2/www/bulma/contact.html"><span>Contact</span></a></p>
            </div>
            </nav>

        </div>

        <div class="column mobile">
            <div class="tile">
                <p class="buttons is-centered">
                    <a class="button">
                        <span class="icon">
                        <i class="fab fa-facebook"></i>
                        </span>
                    </a>
                    <a class="button">
                        <span class="icon">
                        <i class="fab fa-github"></i>
                        </span>
                    </a>
                    <a class="button">
                        <span class="icon">
                        <i class="fab fa-linkedin-in"></i>
                        </span>
                    </a>
                </p>
            </div>
        </div>
        <div class="column contactMain is-10-widescreen is-9-desktop is-paddingless card animated slideInUp">
            <div class="card__corner contactCorner">
                <div class="card__corner-triangle"></div>
            </div>
            <div class="section contact">
                <div class="form column">
            <div class="field">
                <label class="label">Name</label>
                <div class="control has-icons-left has-icons-right">
                    <input class="input" type="text" placeholder="Your name" name="name">
                    <span class="icon is-small is-left"><i class="fas fa-user"></i></span>
                </div>
            </div>

            <div class="field">
                <label class="label">Email</label>
                <div class="control has-icons-left has-icons-right">
                    <input class="input" type="email" placeholder="Your Email" name="email">
                    <span class="icon is-small is-left"><i class="fas fa-envelope"></i></span>
                </div>
            </div>
            <div class="field">
                <label class="label">Message</label>
                <div class="control">
                    <textarea class="textarea" placeholder="Type your message here!" name="message"></textarea>
                </div>
            </div>
            <div class="field">
                <div class="control">
                    <button class="button">Send</button>
                </div>
            </div>
        </div>
            </div>
        </div>
    </div>
</section>
<?PHP
$name = $_POST["name"];
$email = $_POST["email"];
$message = $_POST["message"];
$to = "lalarukh.1992@outlook.com";
$subject = "New Email Address for Mailing List";
$headers = "From: $email\n";
$message = "A visitor to your site has sent the following email address to be added to your mailing list.\n

Email Address: $email;
Name: $name;
Message: $message";

mail($to,$subject,$message,$headers);
echo "Thank you for your message, I will reply to you soon!"
?>
</body>