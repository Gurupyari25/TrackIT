// let passId = document.getElementById("blankML").innerHTML;
// let renderId = document.getElementsByClassName("Row");
// let html = "";
// if (passId == 140001) {
//   Array.from(renderId).forEach((element) => {
//     html += `<td>{{date}}</td>
//             <td>{{140001}}</td>`;
//     element.innerHTML = html;
//   });
// }

let tableML = document.getElementById("MLT");
let tbodyRowCountML = tableML.tBodies[0].rows.length;

if (tbodyRowCountML === 0) {
  document.getElementsByClassName("ML")[0].style.visibility = "hidden";

  let txtML = `<h1>There has not been any class of ML yet</h1> `;
  let parentML = document.getElementById("blankML");
  parentML.innerHTML = txtML;
} else {
  document.getElementById("blankML").style.visibility = "hidden";
}

let tableIOT = document.getElementById("IOTT");
let tbodyRowCountIOT = tableIOT.tBodies[0].rows.length;

if (tbodyRowCountIOT === 0) {
  document.getElementsByClassName("IOT")[0].style.visibility = "hidden";

  let txtIOT = `<h1>There has not been any class of IOT yet</h1> `;
  let parentIOT = document.getElementById("blankIOT");
  parentIOT.innerHTML = txtIOT;
} else {
  document.getElementById("blankIOT").style.visibility = "hidden";
}

let tableCN = document.getElementById("CNT");
let tbodyRowCountCN = tableCN.tBodies[0].rows.length;

if (tbodyRowCountCN === 0) {
  document.getElementsByClassName("CN")[0].style.visibility = "hidden";

  let txtCN = `<h1>There has not been any class of CN yet</h1> `;
  let parentCN = document.getElementById("blankCN");
  parentCN.innerHTML = txtCN;
} else {
  document.getElementById("blankCN").style.visibility = "hidden";
}
