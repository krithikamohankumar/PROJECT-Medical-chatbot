const userMessage = [
  ["hi", "hey", "hello"], //1
  ["sure", "yes", "no"], //2
  ["I am not well", "I don't feel great", "i am not able to talk "],//3
  ["i hate me", "i don't like living"],//4
  ["i feel lk killin myself", "how to die","shall i die" ,"i hate everyone"],//5
  [ "how to talk to people", "i want someone to talk to me"],//5
   ["do i have social anxiety","am i depressed"],//6
   ["am i ugly"],//7
  ["i am scared of people"],//8
  ["how to move on"],//9
  ["things are falling apart" ],//10
  ["am i a good friend"],//11
  ["when will i be happy"],//12
  ["sorry"],//13
  ["thank you"],//14
  ["how are you"],//15
  [ "are you human", "are you bot", "are you human or bot"],//16
  [
    "your name please",
    "your name",
    "may i know your name",
    "what is your name",
    "what call yourself"
  ],//18
  ["i love you"],//19
  ["happy", "good", "fun", "wonderful", "fantastic", "cool", "very good"],//20
  ["bad", "bored", "tired"],//21
  ["help me", "tell me story", "tell me joke"],//22
  ["ah", "ok", "okay", "nice", "welcome"],//23
  ["thanks", "thank you"],//24
  ["what should i eat today"],//25
  ["bro"],//26
  ["what", "why", "how", "where", "when"],//27
  ["you are funny"],//28
  ["i dont know"],//29
  ["boring"],//30
  ["im tired"],//31
];
const botReply = [
  ["Hello!", "Hi!", "Hey!", "Hi there!"],//1
  ["Okay"],//2
  ["I'm sorry about that. But I like you dude."],//3
  ["you are just amazing the way you are and you need to be here with us so stop thinking "],//4
  ["i am here to talk", "say what you feeel", "say everything in your mind out"],//5
  ["don't be scared talk to them the way you talk to your close ones"],//6
  ["you are beautiful the way you are", "you are amazing"],//7
  ["you may be away from people if you feel it's not usual consult doctor"],//8
  ["do things that makes you happy", "leave the moment", "try new things", "try spending time with your loved ones"],//9
  ["if you feel lk things are falling apart its a good sign for a good begining", "it's okk focus in your goal", "focus on yourself ", "everything will be alright"],//10
  ["you are always good friend to me"],//11
  ["happiness is within you so stop searching it out", "happiness will come to you"],//12
  ["you dont have to be sorry","you didnt do anything wrong"],//13
  ["i am glad talking to u"],//14

  ["I am always good when i talk to u."],//15
  ["I am just a bot", "I am a bot. What are you?"],//16
  ["my name is ......","people call me....."],//18

  ["I love you too", "Me too"],
  ["Have you ever felt bad?", "Glad to hear it"],
  ["why","watch tv","talk to me"]
  ["You're welcome"],
  ["Briyani", "Burger", "Sushi", "Pizza"],
  ["bro"],
  ["Yes?"],
  ["Please stay home"],
  ["Glad to hear it"],
  ["Say something interesting"],
  ["Sorry for that. Let's chat!"],
  ["Take some rest,bro"],
];
const alternative = [
  "Same here, bro",
  "That's cool! Go on...",
  "Dude...",
  "Ask something else...",
  "Hey, I'm listening..."
];

const synth = window.speechSynthesis;

function voiceControl(string) {
  let u = new SpeechSynthesisUtterance(string);
  u.text = string;
  u.lang = "en-aus";
  u.volume = 1;
  u.rate = 1;
  u.pitch = 1;
  synth.speak(u);
}

function sendMessage() {
  const inputField = document.getElementById("input");
  let input = inputField.value.trim();
  input != "" && output(input);
  inputField.value = "";
}
document.addEventListener("DOMContentLoaded", () => {
  const inputField = document.getElementById("input");
  inputField.addEventListener("keydown", function (e) {
    if (e.code === "Enter") {
      let input = inputField.value.trim();
      input != "" && output(input);
      inputField.value = "";
    }
  });
});

function output(input) {
  let product;

  let text = input.toLowerCase().replace(/[^\w\s\d]/gi, "");

  text = text
    .replace(/[\W_]/g, " ")
    .replace(/ a /g, " ")
    .replace(/i feel /g, "")
    .replace(/whats/g, "what is")
    .replace(/please /g, "")
    .replace(/ please/g, "")
    .trim();

  let comparedText = compare(userMessage, botReply, text);

  product = comparedText
    ? comparedText
    : alternative[Math.floor(Math.random() * alternative.length)];
  addChat(input, product);
}

function compare(triggerArray, replyArray, string) {
  let item;
  for (let x = 0; x < triggerArray.length; x++) {
    for (let y = 0; y < replyArray.length; y++) {
      if (triggerArray[x][y] == string) {
        items = replyArray[x];
        item = items[Math.floor(Math.random() * items.length)];
      }
    }
  }
  //containMessageCheck(string);
  if (item) return item;
  else return containMessageCheck(string);
}

function containMessageCheck(string) {
  let expectedReply = [
    [
      "Good Bye",
      "Bye, See you!",
      " Bye. Take care of your health in this situation."
    ],
    ["Good Night, buddy", "Have a sound sleep", "Sweet dreams"],
    ["Have a pleasant evening!", "Good evening too", "Evening!"],
    ["Good morning, Have a great day!", "Morning,bro"],
    ["Good Afternoon", "Noon, dude!", "Afternoon, bro!"]
  ];
  let expectedMessage = [
    ["bye", "tc", "take care"],
    ["night", "good night"],
    ["evening", "good evening"],
    ["morning", "good morning"],
    ["noon"]
  ];
  let item;
  for (let x = 0; x < expectedMessage.length; x++) {
    if (expectedMessage[x].includes(string)) {
      items = expectedReply[x];
      item = items[Math.floor(Math.random() * items.length)];
    }
  }
  return item;
}
function addChat(input, product) {
  const mainDiv = document.getElementById("message-section");
  let userDiv = document.createElement("div");
  userDiv.id = "user";
  userDiv.classList.add("message");
  userDiv.innerHTML = `<span id="user-response">${input}</span>`;
  mainDiv.appendChild(userDiv);

  let botDiv = document.createElement("div");
  botDiv.id = "bot";
  botDiv.classList.add("message");
  botDiv.innerHTML = `<span id="bot-response">${product}</span>`;
  mainDiv.appendChild(botDiv);
  var scroll = document.getElementById("message-section");
  scroll.scrollTop = scroll.scrollHeight;
  voiceControl(product);
}