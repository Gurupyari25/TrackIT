openCam = document.getElementById("openCam");
openCam.addEventListener("click", submitForm);

function submitForm() {
  openCam.submit();
}
setTimeout(() => {
  document.getElementById("msg").style.display = "none";
}, 3000);
