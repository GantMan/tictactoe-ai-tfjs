import React from 'react'
import ReactDOM from 'react-dom'
import * as tf from '@tensorflow/tfjs'
import './styles.css'
import { getMoves, getModel, trainOnGames } from './train'
import { saveAs } from 'file-saver'

// TODO: Dis so nasty
const doPredict = async (myBoard, ttt_model) => {
  const tenseBlock = tf.tensor([myBoard])
  const result = await ttt_model.predict(tenseBlock)

  const flatty = result.flatten()
  const maxy = flatty.argMax()
  const move = await maxy.data()
  const allMoves = await flatty.data()

  flatty.dispose()
  tenseBlock.dispose()
  result.dispose()
  maxy.dispose()
  return [move[0], allMoves]
}

function Square(props) {
  const visual = props.value
    ? `square animate__animated animate__flipInX animate__faster ${props.glow}`
    : 'square'
  return (
    <button className={visual} onClick={props.onClick}>
      {props.value}
    </button>
  )
}

const winnerBar = (winner) => {
  if (!winner) return

  return (
    <svg
      className="winLine animate_animated animate__bounceIn"
      width="300"
      height="300"
    >
      <defs>
        <filter id="dropGlow" width="2.5" height="2.5" x="-.15" y="-.15">
          <feGaussianBlur
            id="feGaussianBlur5384"
            in="SourceAlpha"
            stdDeviation="15.000000"
            result="blur"
          />
          <feColorMatrix
            id="feColorMatrix5386"
            result="bluralpha"
            type="matrix"
            values="-1 0 0 0 1 0 -1 0 0 1 0 0 -1 0 1 0 0 0 0.800000 0 "
          />
          <feOffset
            id="feOffset5388"
            in="bluralpha"
            dx="0.000000"
            dy="0.000000"
            result="offsetBlur"
          />
          <feMerge id="feMerge5390">
            <feMergeNode id="feMergeNode5392" in="offsetBlur" />
            <feMergeNode id="feMergeNode5394" in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <line
        x1="20"
        y1="20"
        x2="280"
        y2="280"
        stroke="#fffe"
        stroke-width="5"
        style={{ filter: 'url(#dropGlow)' }}
      ></line>
    </svg>
  )
}

class Board extends React.Component {
  renderSquare(i) {
    const squareVal = this.props.squares[i]
    let glowClass
    if (squareVal === 'X') {
      glowClass = 'red'
    } else if (squareVal) {
      glowClass = 'blue'
    }
    return (
      <Square
        glow={glowClass}
        value={squareVal}
        onClick={() => this.props.onClick(i)}
      />
    )
  }

  render() {
    return (
      <div>
        <div className="board-row">
          {this.renderSquare(0)}
          {this.renderSquare(1)}
          {this.renderSquare(2)}
        </div>
        <div className="board-row">
          {this.renderSquare(3)}
          {this.renderSquare(4)}
          {this.renderSquare(5)}
        </div>
        <div className="board-row">
          {this.renderSquare(6)}
          {this.renderSquare(7)}
          {this.renderSquare(8)}
        </div>
      </div>
    )
  }
}

class Game extends React.Component {
  componentWillUnmount() {
    this.state.activeModel && this.state.activeModel.dispose()
  }

  constructor(props) {
    super(props)
    this.state = {
      games: [],
      history: [
        {
          squares: Array(9).fill(null),
        },
      ],
      stepNumber: 0,
      xIsNext: true,
      activeModel: getModel(),
    }
  }

  about() {
    return (
      <div id="about" class="modal">
        <div class="modal__content">
          <h1>About</h1>
          <div>
            <p className="basic_about">
              This is an exploratory project that uses a regular neural network
              to teach AI how to play Tic Tac Toe. The most common method of
              solving Tic Tac Toe is normally using Q-Learning, but what fun is
              that?
            </p>
            <p>
              By playing an effective 6 or 7 games you can make a pretty
              unbeatable AI!
            </p>
          </div>
          <div>
            <a
              onClick={() =>
                this.state.activeModel.save('downloads://ttt_model')
              }
              class="btn effect01"
            >
              <span>Download Current AI Model</span>
            </a>
            <br />
            <a
              onClick={() => {
                const blob = new Blob(
                  [`{${JSON.stringify(this.state.games, null, 2)}}`],
                  {
                    type: 'application/json;charset=utf-8',
                  }
                )
                saveAs(blob, 'tictactoe.json')
              }}
              class="btn effect01"
            >
              <span>Download Past Games Training Data</span>
            </a>
          </div>
          <br />
          <div class="modal__footer">
            Made with ♥️ by{' '}
            <a href="https://twitter.com/gantlaborde" target="_blank">
              @GantLaborde
            </a>{' '}
            and{' '}
            <a href="https://infinite.red" target="_blank">
              Infinite Red
            </a>
          </div>
          <a href="#" class="modal__close">
            &times;
          </a>
        </div>
      </div>
    )
  }

  handleClick(i) {
    const history = this.state.history.slice(0, this.state.stepNumber + 1)
    const current = history[history.length - 1]
    const squares = current.squares.slice()
    if (calculateWinner(squares) || squares[i]) {
      return
    }
    squares[i] = this.state.xIsNext ? 'X' : 'O'
    this.setState({
      history: history.concat([
        {
          squares: squares,
        },
      ]),
      stepNumber: history.length,
      xIsNext: !this.state.xIsNext,
    })
  }

  async makeAIMove() {
    const history = this.state.history.slice(0, this.state.stepNumber + 1)
    const current = history[history.length - 1]
    const squares = current.squares.slice()

    const AIready = squares.map((v) => {
      if (v === 'X') {
        return this.state.xIsNext ? 1 : -1
      } else if (v === 'O') {
        return this.state.xIsNext ? -1 : 1
      } else {
        return 0
      }
    })
    // console.log(AIready);
    let [move, moves] = await doPredict(AIready, this.state.activeModel)
    // Check if AI made a valid move!
    while (squares[move] !== null && squares.includes(null)) {
      console.log(`AI Failed - Spot ${move} - Resorting to next highest`)
      // Make current move 0
      moves[move] = 0
      move = moves.indexOf(Math.max(...moves))
      // move = Math.floor(Math.random() * 9);
    }

    this.handleClick(move)
  }

  jumpTo(step) {
    const progress =
      step === 0 ? [{ squares: Array(9).fill(null) }] : this.state.history
    this.setState({
      stepNumber: step,
      xIsNext: step % 2 === 0,
      history: progress,
    })
  }

  trainUp(playerLearn) {
    playerLearn = playerLearn || 'O'
    console.log('Train Called - to be more like ', playerLearn)
    // console.log(this.state.history);
    const AllMoves = this.state.history.map((board) => {
      return board.squares.map((v) => {
        if (v === playerLearn) {
          return 1
        } else if (v === null) {
          return 0
        } else {
          return -1
        }
      })
    })

    this.setState(
      (prevState) => {
        const games = prevState.games
        games.push(getMoves(AllMoves))
        return { games }
      },
      () => {
        trainOnGames(this.state.games, (newModel) => {
          window.location.hash = '#'
          this.setState({
            activeModel: newModel,
            stepNumber: 0,
            xIsNext: true,
            history: [
              {
                squares: Array(9).fill(null),
              },
            ],
          })
        })
      }
    )
  }

  render() {
    const history = this.state.history
    const current = history[this.state.stepNumber]
    const winner = calculateWinner(current.squares)

    const moves = history.map((step, move) => {
      const desc = move ? 'Move #' + move : 'Empty Board'
      return (
        <li key={move}>
          <a onClick={() => this.jumpTo(move)} class="btn effect01">
            <span>{desc}</span>
          </a>
        </li>
      )
    })

    let status
    if (winner) {
      status = 'Winner: ' + winner
    } else {
      status = ''
    }

    return (
      <div className="site">
        <div id="training-modal" class="modal">
          <div class="modal__content">
            <h1>Training...</h1>
          </div>
        </div>
        {this.about()}
        <h1 className="animate__animated animate__bounceInDown">
          AI Trainable Tic Tac Toe
        </h1>

        <div className="game">
          {winnerBar(winner)}
          <div className="game-board">
            <Board
              squares={current.squares}
              onClick={(i) => this.handleClick(i)}
            />
          </div>
          <div className="game-info">
            <h3>
              AI has learned from <strong>{this.state.games.length}</strong>{' '}
              game(s)
            </h3>
            <div>
              {status}
              {!winner && (
                <a
                  onClick={() => this.makeAIMove()}
                  class="btn effect01"
                  target="_blank"
                >
                  <span>Make AI Move</span>
                </a>
              )}
            </div>
            <ol>{moves}</ol>
          </div>
        </div>
        <div className="trainSection">
          {(winner || !current.squares.includes(null)) && (
            <a
              href="#training-modal"
              onClick={() => this.trainUp('X')}
              class="btn effect01 animate__animated animate__fadeIn bigx"
            >
              <span>Train AI to play like X</span>
            </a>
          )}
          <br />
          <br />
          {(winner || !current.squares.includes(null)) && (
            <a
              href="#training-modal"
              onClick={() => this.trainUp('O')}
              class="btn effect01 animate__animated animate__fadeIn bigo"
            >
              <span>Train AI to play like O</span>
            </a>
          )}
        </div>
        <div className="footer">
          <div className="footRoof"></div>
          <div className="footerContent">&nbsp;</div>
        </div>
        <p className="about">
          <a href="#about">About this project +</a>
        </p>
      </div>
    )
  }
}

// ========================================

ReactDOM.render(<Game />, document.getElementById('root'))

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ]
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i]
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a]
    }
  }
  return null
}
