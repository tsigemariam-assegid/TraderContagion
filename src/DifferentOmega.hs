module DifferentOmega (diffomega) where
import ABMSimulation
import Control.Monad
import Data.List

diffomega :: IO ()
diffomega = do
    let omega_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 2.0]  
    results <- forM omega_list $ \omega -> do
        putStrLn $ "omega: " ++ show omega
        runABM 1024 0.02178 omega 1 1 1000 10
    return ()


