module AgentBased
    ( Model
    , initializeModel
    , step             -- Function to perform one step of the model
    , dailyReturns     -- Function to get daily returns from the model
    , dailyTradingVolumes -- Function to get daily trading volumes
    , runModel
    ) where


import System.Random (randomRIO)
import System.Random.MWC ( create )
import System.Random.MWC.Distributions (normal)
import Control.Monad (replicateM, replicateM_)
import Data.List (genericLength, sort, mapAccumL)
import Control.Monad.State
    ( MonadState(put, get), MonadIO(liftIO), StateT, execStateT)
import GHC.RTS.Flags (ProfFlags(modSelector))
import Statistics.Sample (mean, stdDev)
import Data.Vector (Vector, fromList, toList)
import Graphics.Gnuplot.Simple
import Graphics.Gnuplot.Advanced


-- import Data.Random ( normal, stdNormal, runRVar)
-- -- Model Parameters
-- n :: Int
-- n = 2 ^ 10

-- t :: Int
-- t = 10000

-- p :: Double
-- p = 0.02178

-- omega, b, size, k, m :: Int
-- omega = 1
-- b = 1
-- size = 1
-- k = 1
-- m = 500

-- d :: Double
-- d = 1.12

-- buy_sell_hold function
buySellHold :: Double -> Int -> IO [Int]
buySellHold p amountTimes = do
    diceRolls <- replicateM amountTimes (randomRIO (0.0, 1.0))
    let indices = filter ((<= (2 * p)) . snd) $ zip [0..] diceRolls
    psis <- mapM (\(idx, _) -> do
                    coin <- randomRIO (0, 1 :: Int)
                    return (idx, if coin == 0 then 1 else -1)
                 ) indices
    return $ foldr (\(idx, val) acc -> take idx acc ++ [val] ++ drop (idx + 1) acc) (replicate amountTimes 0) psis

-- Model data type
data Model = Model {
    n :: Integer,
    p :: Double,
    dailyReturn :: Double,
    tradingVolume :: Int,
    k :: Int,
    omega :: Double,
    dailyReturns :: [Double],
    ct :: Int,
    b :: Int,
    dailyTradingVolumes :: [Int]
} deriving (Show)

mean' :: Model -> Double
mean' model = (fromIntegral (n model) / abs (dailyReturn model)) ** (omega model)


distributeOpinionGroups :: Model -> IO Int
distributeOpinionGroups model
    | b model == 0 = return $ round $ mean' model
    | abs (dailyReturn model) >= fromIntegral (n model) = return 1
    | otherwise = do
        let mean = mean' model
            stdDev = sqrt (mean * fromIntegral (b model))
            minValue = mean - stdDev
            maxValue = mean + stdDev
        -- liftIO $ putStrLn $ show mean 
        -- g <- create
        -- c <- runRVar (normal mean stdDev) g
        g <- create
        c <- normal mean stdDev g
        -- liftIO $ putStrLn $ "c: " ++ show c ++ show mean ++ show stdDev
        let d = max 1 (round c)
        -- let d = if c <= 0 then 1 else round c
        return $ min d (fromIntegral (n model))
applyBoundaries :: Double -> Double -> Double -> Double
applyBoundaries dailyReturn minReturn maxReturn =
    -- | abs dailyReturn < minReturn = signum dailyReturn * minReturn
    -- | abs dailyReturn > maxReturn = signum dailyReturn * maxReturn
    -- | otherwise = dailyReturn
    let sign = if dailyReturn < 0 then -1 else 1
    in sign * (min maxReturn (max minReturn (abs dailyReturn)))

step :: StateT Model IO Int
step = do
    model <- get
    c <- liftIO $ distributeOpinionGroups model
    psis <- liftIO $ buySellHold (p model) c
    let averageAgentsPerGroup =  fromIntegral (n model) / fromIntegral c
        returnMatrix = map ((* averageAgentsPerGroup) . fromIntegral) psis
    -- liftIO $ putStrLn $ "c: " ++ show c ++ ", avgAgentsPerGroup: " ++ show averageAgentsPerGroup ++ ", returnMatrix: " ++ show returnMatrix
    let
        tradingVolume = round $ sum $ map abs returnMatrix
        dailyReturn' = sum returnMatrix  -- Should be Double now
        minimumReturn = fromIntegral (n model) ** ((omega model - 1) / (omega model))
        dailyReturn'' = applyBoundaries dailyReturn' minimumReturn (fromIntegral (n model))
    put model { dailyReturn = dailyReturn'',
                dailyReturns = dailyReturns model ++ [dailyReturn''],
                dailyTradingVolumes = dailyTradingVolumes model ++ [tradingVolume],
                ct = ct model + 1 }
    return $ ct model + 1

runModel :: Int -> Model -> IO Model
runModel t model = execStateT (replicateM_ t step) model

standardScale :: [Double] -> [Double]
standardScale xs = map (\x -> (x - m) / s) absXs
  where
    absXs = map abs xs  -- Take the absolute value of each element
    vXs = fromList absXs  -- Convert the list to a Vector
    m = mean vXs  -- Calculate the mean
    s = stdDev vXs  -- Calculate the standard deviation
initializeModel :: Integer -> Double -> Double -> Int -> Int -> Model
initializeModel nVal pVal omegaVal bVal kVal = Model {
        n = nVal,
        p = pVal,
        dailyReturn = 1.0,
        dailyReturns = [],
        dailyTradingVolumes = [],
        omega = omegaVal,
        b = bVal,
        k = kVal,
        tradingVolume = 0,
        ct = 0
        }

-- Generate logarithmically spaced bins
-- logspace :: Floating a => a -> a -> Int -> [a]
-- logspace start stop num = map (exp . (start +) . (* step)) [0..fromIntegral (num - 1)]
--     where
--       step = (stop - start) / fromIntegral (num - 1)
logspace :: Floating a => a -> a -> Int -> [a]
logspace start stop num =
  let step = (stop - start) / fromIntegral (num - 1)
  in [exp (start + fromIntegral i * step) | i <- [0..num - 1]]

-- Compute the cumulative histogram data
-- This function will return the bin value and the count of elements greater than or equal to the bin value
cumulativeHistogram :: (Ord a, Num b, Enum b) => [a] -> [a] -> [(a, b)]
cumulativeHistogram datalist bins = reverse $ snd $ mapAccumL countGreater (sort datalist) (reverse bins)
    where
      countGreater sortedData bin = (dropWhile (< bin) sortedData, (bin, fromIntegral $ length sortedData))

-- Example data and bins
-- let scaledAbsABMReturns = ...
-- let t = ...
-- let bins = logspace 0 (log 5) t
-- let histData = cumulativeHistogram scaledAbsABMReturns bins

-- Plotting with Gnuplot (if you're using the Gnuplot library)
plotLogLog :: [(Double, Double)] -> IO ()
plotLogLog = plotListStyle [Title "Probability distribution of absolute returns (log-log)",
                                      XLabel "Absolute daily returns",
                                      YLabel "P(|absolute returns| > x)",
                                      Custom "set logscale x 10" [],
                                      Custom "set logscale y 10" [],
                                      Custom "set ouput" ["/Users/tsigemariam/TraderContagion/src/data/ouput.png"]]
                                     (defaultStyle {plotType = Boxes})

-- logLogAxis :: Axis B V2 Double
-- logLogAxis = r2Axis &~ do
--   linePlot' $ map p2 points
--   setAxisStyles
--   withAxis $ do
--     set (axScale .~ LogLog) . axisScale
--     xAxis . axisLabel .= "Absolute daily returns"
--     yAxis . axisLabel .= "P(|absolute returns| > x)"

-- setAxisStyles :: PlotMonad t Double m => m ()
-- setAxisStyles = do
--   xAxis . axisTicks . majorTicks . tickLength .= small
--   yAxis . axisTicks . majorTicks . tickLength .= small
--   xAxis . axisGridStyle .= dashedLine 0.9 (sRGB24read "#bbbbbb")
-- .=  yAxis . axisGridStyle .= dashedLine 0.9 (sRGB24read "#bbbbbb")
--   xAxis . axisLine .= Just (solidLine 1 (sRGB24read "#000000"))
--   yAxis . axisLine  Just (solidLine 1 (sRGB24read "#000000"))

-- plotData :: [(Double, Double)] -> Renderable ()
-- plotData histogramData = toRenderable $ do
--     layout_title .= "Log-Log Cumulative Histogram"
--     layout_x_axis . laxis_title .= "Log(X)"
--     layout_x_axis . laxis_log .~ True  -- Logarithmic X-axis
--     layout_y_axis . laxis_title .= "Log(Cumulative Count)"
--     layout_y_axis . laxis_log .~ True  -- Logarithmic Y-axis
--     plot $ line "Cumulative Histogram" [histogramData]
--     layout_y_axis . laxis_override .= axisGridHide  -- Hide grid lines on the y-axis

main :: IO ()
main = do
    -- Example usage of buySellHold
    -- decisions <- buySellHold 0.2 10
    -- print decisions
    -- print $ length decisions == 10
    let initialmodel = Model {n = 1024,  p = 0.02178, dailyReturn = 1.0, dailyReturns = [], dailyTradingVolumes = [], omega = 1, b = 1, k = 1, tradingVolume = 0, ct = 0}
    finalmodel <- runModel 10000 initialmodel
    let y = standardScale (dailyReturns finalmodel)
        points = zip ([1..] :: [Int]) y
    plotList [] points
        -- bins = logspace 0 (log 5) 10000
    --     logy = [log(j) | j <- y]
    --     histData = cumulativeHistogram logy bins
    --     points = zip ([1..] :: [Double]) y
    --     transformedData = [(log x, log y) | (x, y) <- points]
    -- plotPathsStyle [] [(defaultStyle {plotType = Points}, histData)]
    -- plotPathsStyle [] [(defaultStyle {plotType = Boxes}, histData)]
    -- renderableToFile def "log_log_plot.png" plotData  histData
    --     points = zip ([1..] :: [Int]) y
    -- liftIO $ putStrLn "plotting"
    -- plotPathsStyle [] [(defaultStyle {plotType = Boxes}, histData)]
    -- mainWith $ renderAxis logLogAxis


-- n :: Int
-- n = 2 ^ 10

-- t :: Int
-- t = 10000

-- p :: Double
-- p = 0.02178

-- omega, b, size, k, m :: Int
-- omega = 1
-- b = 1
-- size = 1
-- k = 1
-- m = 500

-- d :: Double
-- d = 1.12