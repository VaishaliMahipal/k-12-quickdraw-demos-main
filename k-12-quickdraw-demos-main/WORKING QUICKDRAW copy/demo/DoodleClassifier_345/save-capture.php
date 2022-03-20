<?php
define('UPLOAD_DIR', 'images2/');  
$image = $_POST["image"];
$image=explode(";",$image)[1];
$image = explode(",",$image)[1];
$image=str_replace(" ","+",$image);
$image= base64_decode($image);
$filename = UPLOAD_DIR . uniqid() . '.jpeg';  
file_put_contents($filename,$image);

echo "Done";




