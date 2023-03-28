const table = document.querySelector("table");
const tbodyRowCount = table.tBodies[0].rows.length;

if (tbodyRowCount === 0) {
  document.getElementsByClassName("container")[0].style.display = "none";

  var txt = `<h1>You haven't taken any class yet!</h1> `;
  var parent = document.getElementById("blank");
  parent.innerHTML = txt;
} else {
  document.getElementById("blank").style.display = "none";
}
