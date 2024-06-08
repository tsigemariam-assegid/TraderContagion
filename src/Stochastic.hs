module Stochastic
    (
        StochasticModel(..)
        , runModel 
        , initializeStochasticModel
    ) where


import System.Random (randomIO, Random)
import Data.List (genericLength)
import System.Random.MWC
import System.Random.MWC.Distributions (normal)
import Control.Monad 
import Control.Monad.State
    ( MonadState(put, get), MonadIO(liftIO), StateT, execStateT)
import Statistics.Sample (mean, stdDev)
import Data.Vector (Vector, fromList, toList)
import Graphics.Gnuplot.Simple (plotList)
data StochasticModel = StochasticModel {
    n :: Integer,
    p :: Double,
    initial :: Double,
    returns :: [Double],
    time_horizon :: Bool,
    d :: Double, 
    m :: Int
} deriving (Show)

initializeStochasticModel :: Integer -> Double -> Double -> Bool -> Double -> Int -> StochasticModel
initializeStochasticModel nVal pVal initialVal timeHorizonVal dVal mVal = StochasticModel {
    n = nVal,
    p = pVal,
    initial = initialVal,
    returns = [initialVal],    
    time_horizon = timeHorizonVal,
    d = dVal,
    m = mVal
}

-- Function to calculate time horizons
timeHorizons :: StochasticModel -> Double
timeHorizons model = sum timeHorizonsList / sum alphaList
  where
    returnsList = returns model
    mValue = m model
    dValue = d model
    timeHorizonsList = [ (fromIntegral i ** (-dValue)) * absReturn i | i <- [1..mValue] ]
    alphaList = [ fromIntegral i ** (-dValue) | i <- [1..mValue] ]
    absReturn i
      | length returnsList == 1 = (abs (head returnsList))
      | i >= length returnsList = (abs (head returnsList - last returnsList))
      | otherwise = (abs (last returnsList - (returnsList !! (length returnsList - i))))

-- Function to perform a step
step :: StochasticModel -> IO StochasticModel
step model = do
    g <- createSystemRandom
    normalVal <- normal 0.0 1.0 g
    -- liftIO $ putStrLn $ show normalVal 
    let variance = if time_horizon model
                   then 2 * p model * fromIntegral (n model) * timeHorizons model
                   else 2 * p model * fromIntegral (n model) *  (abs (last (returns model)))
    let std = sqrt variance
    let value = std * normalVal
    let newReturns = returns model ++ [value]
    return model { returns = newReturns }
 
runModel :: (Eq t, Num t) => t -> StochasticModel -> IO StochasticModel
runModel t model = iterateM t model
  where
    iterateM 0 m = return m
    iterateM n m = step m >>= \newModel -> iterateM (n-1) newModel


standardScale :: [Double] -> [Double]
standardScale xs = map (\x -> (x - m) / s) absXs
  where
    absXs = map abs xs  -- Take the absolute value of each element
    vXs = fromList absXs  -- Convert the list to a Vector
    m = mean vXs  -- Calculate the mean
    s = stdDev vXs  -- Calculate the standard deviation

-- main :: IO ()
-- main = do
--     let stochastic = StochasticModel {n = 1024,  p = 0.02178, initial = 10.0, time_horizon = False, m = 10, d = 1.12, returns = [10.0]}
--     let stochasticHorizons = StochasticModel {n = 1024, initial = 10.0, p = 0.02178, time_horizon = True, m = 10, d = 1.12, returns = [10.0]}
--     finalStochastic <- runModel 10000 stochastic 
--     finalStochasticHorizons <- runModel 10000 stochasticHorizons
--     let y = standardScale (returns finalStochasticHorizons)
--         points = zip ([1..] :: [Int]) y
--     plotList [] points
--     return 
