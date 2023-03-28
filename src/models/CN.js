const mongoose = require("mongoose");
const { mlSchema, MLstatus } = require("./ML");

// mongoose
//   .connect("mongodb://localhost:27017/trackit", {
//     useNewUrlParser: true,
//     useUnifiedTopology: true,
//   })
//   .then(() => {
//     console.log("Connection successful!");
//   })
//   .catch((err) => {
//     console.log(err);
//   });

const CNstatus = new mongoose.model("CNstatus", mlSchema);

// const newCN = async () => {
//   try {
//     const newStatus = new CNstatus({
//       01: "Present",
//       04: "Present",
//       05: "Present",
//     });
//     const result = await newStatus.save();
//     console.log(result);
//   } catch (err) {
//     console.log(err);
//   }
// };
// newCN();

module.exports = CNstatus;
