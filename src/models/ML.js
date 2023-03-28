const mongoose = require("mongoose");
const moment = require("moment");

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

const mlSchema = new mongoose.Schema({
  Date: {
    type: String,
    default: moment(new Date()).format("DD/MM/YYYY"),
    index: { unique: true, sparse: true },
    required: true,
  },
  140001: {
    type: String,
    default: "Absent",
    required: true,
  },
  140002: {
    type: String,
    default: "Absent",
    required: true,
  },
  140003: {
    type: String,
    default: "Absent",
    required: true,
  },
  140004: {
    type: String,
    default: "Absent",
    required: true,
  },
  140005: {
    type: String,
    default: "Absent",
    required: true,
  },
});

const MLstatus = new mongoose.model("MLstatus", mlSchema);

// const newMl = async () => {
//   try {
//     const newStatus = new MLstatus({
//       date: "17/06/2022",
//       140001: "Present",
//     });
//     const result = await newStatus.save();
//     console.log(result);
//   } catch (err) {
//     console.log(err);
//   }
// };
// newMl();

module.exports = { mlSchema, MLstatus };
