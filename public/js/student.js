setTimeout(() => {
  document.getElementById("msg").style.display = "none";
}, 3000);

setTimeout(() => {
  document.getElementById("err").style.display = "none";
}, 3000);

let train = document.getElementById("train");
console.log(train.className);
if (train.className == "hide") {
  train.style.display = "none";
} else {
  train.style.display = "block";
}
